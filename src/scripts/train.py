import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import time
import argparse
from tqdm import tqdm
from collections import Counter

from bms.dataset import collate_fn, BMSDataset, SequentialSampler
from bms.transforms import get_train_transforms, get_val_transforms
from bms.model import EncoderCNN, DecoderWithAttention
from bms.metrics import AverageMeter, get_levenshtein_score, get_accuracy
from bms.model_config import model_config
from bms.utils import load_pretrain_model, make_dir, FilesLimitControl, sec2min


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SCALER = torch.cuda.amp.GradScaler()

torch.backends.cudnn.benchmark = True


def get_sample_probs(loss_no_reduction, decode_lengths):
    sample_probs = []
    s = 0
    for length in decode_lengths:
        sample_probs.append(
            loss_no_reduction[s:s+length].sum().item()
        )
        s = s + length
    return sample_probs


def get_token_weights(data_csv):
    tokens_indexes = data_csv['Tokens_indexes'].values
    token_index2count = Counter(c for clist in tokens_indexes for c in clist)

    token_index2weights = {}
    for token_index, count in token_index2count.items():
        token_index2weights[token_index] = 1 + 5 * ((1/count) ** 0.25)

    token_weights = [token_index2weights[idx]
                     for idx in range(len(token_index2weights))]
    print('min class weight:', min(token_weights))
    print('max class weight:', max(token_weights))
    return torch.FloatTensor(token_weights).to(DEVICE)


def train_loop(data_loader, encoder, decoder, criterion, optimizer,
               sampler, criterion_no_reduction):
    loss_avg = AverageMeter()
    strat_time = time.time()
    len_avg = AverageMeter()
    if not decoder.training:
        decoder.train()
    if not encoder.training:
        encoder.train()

    tqdm_data_loader = tqdm(data_loader, total=len(data_loader), leave=False)
    for images, captions, lengths, text_true, idxs in tqdm_data_loader:
        decoder.zero_grad()
        encoder.zero_grad()

        images = images.to(DEVICE)
        captions = captions.to(DEVICE)
        batch_size = len(text_true)
        with torch.cuda.amp.autocast():
            features = encoder(images)
            predictions, decode_lengths = decoder(features, captions, lengths)
            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = captions[:, 1:]
            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]
            outputs = pack_padded_sequence(predictions, decode_lengths, batch_first=True)[0]
            loss = criterion(outputs, targets)  # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
            loss_no_reduction = criterion_no_reduction(outputs, targets)

        # update sample probs in batchsampler
        sample_probs = get_sample_probs(loss_no_reduction, decode_lengths)
        sampler.update_sample_probs(sample_probs, idxs)

        loss_avg.update(loss.item(), batch_size)
        mean_len = lengths.sum().item() / len(lengths)
        len_avg.update(mean_len, batch_size)
        SCALER.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(),
                                       model_config['encoder_clip_grad_norm'])
        torch.nn.utils.clip_grad_norm_(decoder.parameters(),
                                       model_config['decoder_clip_grad_norm'])
        SCALER.step(optimizer)
        SCALER.update()
    loop_time = sec2min(time.time() - strat_time)
    return loss_avg.avg, len_avg.avg, loop_time


def val_loop(data_loader, encoder, decoder, tokenizer, max_seq_length):
    levenshtein_avg = AverageMeter()
    acc_avg = AverageMeter()
    strat_time = time.time()
    if decoder.training:
        decoder.eval()
    if encoder.training:
        encoder.eval()

    tqdm_data_loader = tqdm(data_loader, total=len(data_loader), leave=False)
    for images, _, _, text_true, _ in tqdm_data_loader:
        images = images.to(DEVICE)
        batch_size = len(text_true)
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                features = encoder(images)
                predictions = decoder.predict(
                    features, max_seq_length, tokenizer.token2idx["<sos>"])
        predicted_sequence = torch.argmax(predictions.detach().cpu(), -1).numpy()
        text_preds = tokenizer.predict_captions(predicted_sequence)
        levenshtein_avg.update(get_levenshtein_score(text_true, text_preds), batch_size)
        acc_avg.update(get_accuracy(text_true, text_preds), batch_size)
    loop_time = sec2min(time.time() - strat_time)
    return levenshtein_avg.avg, acc_avg.avg, loop_time


def get_loaders(args, data_csv_train, data_csv_val):
    # create train dataset and dataloader
    print(f'Val dataset len: {data_csv_val.shape[0]}')
    if args.epoch_size is not None:
        print(f'Train dataset len: {args.epoch_size}')
    else:
        print(f'Train dataset len: {data_csv_train.shape[0]}')

    train_transform = get_train_transforms(model_config['image_height'],
                                           model_config['image_width'],
                                           args.transf_prob)
    train_dataset = BMSDataset(data_csv_train, train_transform)
    # create train dataloader with custom batch_sampler
    if args.sample_probs_csv_path:
        df = pd.read_csv(args.sample_probs_csv_path)
        init_sample_probs = df['sample_probs'].values
        print('Load sample probs from csv')
    else:
        init_sample_probs = None
    sampler = SequentialSampler(
        dataset=data_csv_train,
        init_sample_probs=init_sample_probs,
        dataset_len=args.epoch_size,
        sample_probs_power=args.sample_probs_power
    )
    batcher = torch.utils.data.BatchSampler(
        sampler, batch_size=args.train_batch_size, drop_last=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=batcher,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        collate_fn=collate_fn
    )

    # create val dataset and dataloader
    val_transform = get_val_transforms(model_config['image_height'],
                                       model_config['image_width'])
    val_dataset = BMSDataset(data_csv_val, val_transform)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    return train_loader, val_loader, sampler


def main(args):
    train_csv = pd.read_pickle(model_config["paths"]["train_csv"])
    val_csv = pd.read_pickle(model_config["paths"]["val_csv"])
    max_seq_length = train_csv['Tokens_len'].max()
    print(max_seq_length)

    # Create model directory
    make_dir(args.model_path)

    tokenizer = torch.load(model_config["paths"]["tokenizer"])
    train_loader, val_loader, sampler = get_loaders(args, train_csv, val_csv)

    # Build the models
    encoder = EncoderCNN()
    if args.encoder_pretrain:
        states = torch.load(args.encoder_pretrain, map_location=DEVICE)
        encoder.load_state_dict(states)
        print('Load pretrained encoder')
    encoder.to(DEVICE)
    decoder = DecoderWithAttention(
        attention_dim=model_config['attention_dim'],
        embed_dim=model_config['embed_dim'],
        decoder_dim=model_config['decoder_dim'],
        vocab_size=len(tokenizer),
        device=DEVICE,
        dropout=model_config['dropout'],
    )
    if args.decoder_pretrain:
        states = load_pretrain_model(args.decoder_pretrain, decoder, DEVICE)
        decoder.load_state_dict(states)
        print('Load pretrained decoder')
    decoder.to(DEVICE)

    # Loss and optimizer
    # class_weigths = get_token_weights(train_csv)
    criterion = nn.CrossEntropyLoss()
    criterion_no_reduction = nn.CrossEntropyLoss(reduction='none')
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                  mode='max',
                                  factor=args.ReduceLROnPlateau_factor,
                                  patience=args.ReduceLROnPlateau_patience)

    encoder_limit_control = FilesLimitControl()
    decoder_limit_control = FilesLimitControl()
    sample_limit_control = FilesLimitControl()
    best_acc = -np.inf

    levenshtein_avg, acc_avg, loop_time = val_loop(
        val_loader, encoder, decoder, tokenizer, max_seq_length)
    print(f'Validation, Levenshtein: {levenshtein_avg:.4f}, '
          f'acc: {acc_avg:.4f}, loop_time: {loop_time}')
    for epoch in range(10000):
        loss_avg, len_avg, loop_time = \
            train_loop(train_loader, encoder, decoder, criterion, optimizer,
                       sampler, criterion_no_reduction)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print(f'\nEpoch {epoch}, Loss: {loss_avg:.4f}, '
              f'Avg seq length: {len_avg:.2f}, '
              f'Samples prob greater 1: {(sampler.init_sample_probs > 1).sum()}, '
              f'Unique samples in epoch: {len(np.unique(sampler.dataset_indexes))}, '
              f'LR: {lr:.7f}, loop_time: {loop_time}')

        levenshtein_avg, acc_avg, loop_time = val_loop(
            val_loader, encoder, decoder, tokenizer, max_seq_length)
        print(f'Validation, Levenshtein: {levenshtein_avg:.4f}, '
              f'acc: {acc_avg:.4f}, loop_time: {loop_time}')

        scheduler.step(acc_avg)

        if acc_avg > best_acc:
            best_acc = acc_avg
            encoder_save_path = os.path.join(
                args.model_path, f'encoder-{epoch}-{acc_avg:.4f}.ckpt')
            decoder_save_path = os.path.join(
                args.model_path, f'decoder-{epoch}-{acc_avg:.4f}.ckpt')
            torch.save(encoder.state_dict(), encoder_save_path)
            torch.save(decoder.state_dict(), decoder_save_path)
            print('Val weights saved')
            encoder_limit_control(encoder_save_path)
            decoder_limit_control(decoder_save_path)

            probs_csv_save_path = os.path.join(
                args.model_path, f'sample_probs-{epoch}-{acc_avg:.4f}.csv')
            probs_csv = pd.DataFrame(data=sampler.init_sample_probs,
                                     columns=["sample_probs"])
            probs_csv.to_csv(probs_csv_save_path, index=False)
            sample_limit_control(probs_csv_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='/workdir/data/experiments/test/',
                        help='Path for saving trained models')
    parser.add_argument('--encoder_pretrain', type=str, default='',
                        help='Encoder pretrain path')
    parser.add_argument('--decoder_pretrain', type=str, default='',
                        help='Decoder pretrain path')
    parser.add_argument('--sample_probs_csv_path', type=str, default='',
                        help='Path to csv with samples probs for batch sampler')
    parser.add_argument('--train_batch_size', type=int, default=50)
    parser.add_argument('--val_batch_size', type=int, default=256)
    parser.add_argument('--epoch_size', type=int, default=200000)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--prefetch_factor', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--transf_prob', type=float, default=0.25)
    parser.add_argument('--sample_probs_power', type=float, default=0.15,
                        help='The degree to which the sample probs is raised \
                              to make probs smoother/sharper.')
    parser.add_argument('--ReduceLROnPlateau_factor', type=float, default=0.8)
    parser.add_argument('--ReduceLROnPlateau_patience', type=int, default=5)
    parser.add_argument('--loss_threshold_to_validate', type=float, default=0.05)

    args = parser.parse_args()
    main(args)
