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
from bms.metrics import (
    AverageMeter, get_levenshtein_score, get_accuracy, sec2min
)
from bms.model_config import model_config
from bms.utils import load_pretrain_model


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SCALER = torch.cuda.amp.GradScaler()

torch.backends.cudnn.benchmark = True


def train_loop(args, data_loader, encoder, decoder, criterion, optimizer):
    loss_avg = AverageMeter()
    strat_time = time.time()
    if not decoder.training:
        decoder.train()
    if not encoder.training:
        encoder.train()

    tqdm_data_loader = tqdm(data_loader, total=len(data_loader), leave=False)
    for images, captions, lengths, _ in tqdm_data_loader:
        decoder.zero_grad()
        encoder.zero_grad()

        images = images.to(DEVICE)
        captions = captions.to(DEVICE)

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

        loss_avg.update(loss.item(), args.train_batch_size)
        SCALER.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(),
                                       model_config['encoder_clip_grad_norm'])
        torch.nn.utils.clip_grad_norm_(decoder.parameters(),
                                       model_config['decoder_clip_grad_norm'])
        SCALER.step(optimizer)
        SCALER.update()
    loop_time = sec2min(time.time() - strat_time)
    return loss_avg.avg, loop_time


def val_loop(args, data_loader, encoder, decoder, tokenizer, max_seq_length):
    levenshtein_avg = AverageMeter()
    acc_avg = AverageMeter()
    strat_time = time.time()
    if decoder.training:
        decoder.eval()
    if encoder.training:
        encoder.eval()

    tqdm_data_loader = tqdm(data_loader, total=len(data_loader), leave=False)
    for images, _, _, text_true in tqdm_data_loader:
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

    # Make FOLDER_2_FREQ dict for batch sampler.
    # Make long sequences more frequent in batches, but not to
    # make rare samples occurred too often to not to overtrain the model.
    FOLDER_2_FREQ = {}
    min_len = min(data_csv_train['Tokens_len'].values)
    max_len = max(data_csv_train['Tokens_len'].values)
    len2samples = Counter(data_csv_train['Tokens_len'].values)
    total_samples = sum(len2samples.values())
    for i in range(min_len, max_len+1):
        FOLDER_2_FREQ[i] = 1 + (i**2) * (len2samples[i] / total_samples)

    train_transform = get_train_transforms(model_config['image_height'],
                                           model_config['image_width'],
                                           args.transf_prob)
    train_dataset = BMSDataset(
        data_csv=data_csv_train,
        transform=train_transform
    )
    # create train dataloader with custom batch_sampler
    sampler = SequentialSampler(
        data_csv_train, FOLDER_2_FREQ, args.epoch_size, args.train_batch_size)
    batcher = torch.utils.data.BatchSampler(
        sampler, batch_size=args.train_batch_size, drop_last=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=batcher,
        num_workers=args.num_workers,
        prefetch_factor=4,
        collate_fn=collate_fn
    )

    # create val dataset and dataloader
    val_transform = get_val_transforms(
        model_config['image_height'], model_config['image_width'])
    val_dataset = BMSDataset(
        data_csv=data_csv_val,
        transform=val_transform
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    return train_loader, val_loader


def main(args):
    train_csv = pd.read_pickle('/workdir/data/processed/train_labels_processed.pkl')
    val_csv = pd.read_pickle('/workdir/data/processed/val_labels_processed.pkl')
    max_seq_length = train_csv['Tokens_len'].max()
    print(max_seq_length)

    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    tokenizer = torch.load('/workdir/data/processed/tokenizer.pth')
    train_loader, val_loader = get_loaders(args, train_csv, val_csv)

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
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer,
                                  factor=args.ReduceLROnPlateau_factor,
                                  patience=args.ReduceLROnPlateau_patience)

    best_loss = np.inf
    best_loss_for_validation = np.inf
    saved_weights_paths = []

    for epoch in range(10000):
        loss_avg, loop_time = train_loop(args, train_loader, encoder, decoder,
                                         criterion, optimizer)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print('Epoch {}, Loss: {:.4f}, LR: {:.7f}, loop_time: {}'.format(
            epoch, loss_avg, lr, loop_time))

        scheduler.step(loss_avg)

        if loss_avg < best_loss:
            best_loss = loss_avg
            print('Val weights saved')
            encoder_save_path = os.path.join(
                args.model_path,
                'encoder-{}-{:.4f}.ckpt'.format(epoch, loss_avg))
            decoder_save_path = os.path.join(
                args.model_path,
                'decoder-{}-{:.4f}.ckpt'.format(epoch, loss_avg))
            torch.save(encoder.state_dict(), encoder_save_path)
            torch.save(decoder.state_dict(), decoder_save_path)
            saved_weights_paths.append(
                (encoder_save_path, decoder_save_path)
            )
            if len(saved_weights_paths) > args.max_weights_to_save:
                old_weights_paths = saved_weights_paths.pop(0)
                for old_weights_path in old_weights_paths:
                    if os.path.exists(old_weights_path):
                        os.remove(old_weights_path)
                        print(f"Model removed '{old_weights_path}'")
        if loss_avg < best_loss_for_validation * (1 - args.loss_threshold_to_validate):
            best_loss_for_validation = loss_avg
            levenshtein_avg, acc_avg, loop_time = val_loop(
                args, val_loader, encoder, decoder, tokenizer, max_seq_length)
            print('\n Validation, Levenshtein: {:.4f}, acc: {:.4f}, loop_time: {}'.format(
                levenshtein_avg, acc_avg, loop_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='/workdir/data/experiments/test/',
                        help='Path for saving trained models')
    parser.add_argument('--encoder_pretrain', type=str, default='',
                        help='Encoder pretrain path')
    parser.add_argument('--decoder_pretrain', type=str, default='',
                        help='Decoder pretrain path')
    parser.add_argument('--train_batch_size', type=int, default=200)
    parser.add_argument('--val_batch_size', type=int, default=512)
    parser.add_argument('--epoch_size', type=int, default=200000)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--transf_prob', type=float, default=0.25)
    parser.add_argument('--ReduceLROnPlateau_factor', type=float, default=0.7)
    parser.add_argument('--ReduceLROnPlateau_patience', type=int, default=10)
    parser.add_argument('--max_weights_to_save', type=int, default=3)
    parser.add_argument('--loss_threshold_to_validate', type=float, default=0.1)

    args = parser.parse_args()
    main(args)
