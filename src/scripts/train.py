import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import pandas as pd
import os
import argparse
import time

from bms.dataset import collate_fn, BMSDataset
from bms.transforms import get_train_transforms, get_val_transforms
from bms.model import EncoderCNN, DecoderWithAttention
from bms.metrics import AverageMeter, time_remain, get_score
from bms.utils import load_pretrain_model


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_loop(
    args, data_loader, encoder, decoder, criterion, optimizer, epoch, scheduler
):
    total_step = len(data_loader)
    loss_avg = AverageMeter()
    start = time.time()
    if not decoder.training:
        decoder.train()
    if not encoder.training:
        encoder.train()

    for i, (images, captions, lengths, _) in enumerate(data_loader):
        decoder.zero_grad()
        encoder.zero_grad()
        # Set mini-batch dataset
        images = images.to(DEVICE)
        captions = captions.to(DEVICE)
        # Forward, backward and optimize
        features = encoder(images)
        predictions, decode_lengths = decoder(features, captions, lengths)
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = captions[:, 1:]
        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]
        outputs = pack_padded_sequence(predictions, decode_lengths, batch_first=True)[0]
        loss = criterion(outputs, targets)  # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_avg.update(loss.item(), args.batch_size)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5)
        optimizer.step()
        # Print log info
        if i % args.log_step == 0 and i > 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time: {}'
                  .format(epoch, args.num_epochs, i, total_step, loss_avg.avg,
                          time_remain(start, (i+1)/total_step)))
    # Save the model checkpoints
    scheduler.step()
    torch.save(decoder.state_dict(), os.path.join(
        args.model_path,
        'decoder-{}-{:.4f}.ckpt'.format(epoch+1, loss_avg.avg)))
    torch.save(encoder.state_dict(), os.path.join(
        args.model_path,
        'encoder-{}-{:.4f}.ckpt'.format(epoch+1, loss_avg.avg)))


def val_loop(args, data_loader, encoder, decoder, criterion, tokenizer, max_seq_length):
    acc_avg = AverageMeter()
    if decoder.training:
        decoder.eval()
    if encoder.training:
        encoder.eval()

    for i, (images, captions, lengths, text_true) in enumerate(data_loader):
        # Set mini-batch dataset
        images = images.to(DEVICE)
        captions = captions.to(DEVICE)
        # targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
        # Forward
        with torch.no_grad():
            features = encoder(images)
            predictions = decoder.predict(features, max_seq_length, tokenizer)
        predicted_sequence = torch.argmax(predictions.detach().cpu(), -1).numpy()
        text_preds = tokenizer.predict_captions(predicted_sequence)
        acc_avg.update(get_score(text_true, text_preds), args.batch_size)
    # Print log info
    print('Val step Acc: {:.4f}'.format(acc_avg.avg))


def get_loaders(args, data_csv):
    train_data_size = int(0.98 * len(data_csv))
    data_csv_train = data_csv.iloc[:train_data_size, :]
    data_csv_val = data_csv.iloc[train_data_size:, :]
    if args.train_dataset_len is not None:
        print(f'Train dataset len: {args.train_dataset_len}')
    else:
        print(f'Train dataset len: {data_csv_train.shape[0]}')
    print(f'Train dataset len: {data_csv_val.shape[0]}')

    train_transform = get_train_transforms(args.output_height, args.output_width)
    val_transform = get_val_transforms(args.output_height, args.output_width)
    train_dataset = BMSDataset(
        data_csv=data_csv_train,
        restrict_dataset_len=args.train_dataset_len,
        transform=train_transform
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    val_dataset = BMSDataset(
        data_csv=data_csv_val,
        transform=val_transform
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    return train_loader, val_loader


def main(args):
    data_csv = pd.read_pickle('/workdir/data/processed/train_labels_processed.pkl')
    max_seq_length = data_csv['InChI_index'].map(len).max()

    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    tokenizer = torch.load('/workdir/data/processed/tokenizer.pth')
    train_loader, val_loader = get_loaders(args, data_csv)

    # Build the models
    encoder = EncoderCNN()
    if args.encoder_pretrain:
        states = torch.load(args.encoder_pretrain, map_location=DEVICE)
        encoder.load_state_dict(states)
        print('Load pretrained encoder')
    encoder.to(DEVICE)
    decoder = DecoderWithAttention(
        attention_dim=args.attention_dim,
        embed_dim=args.embed_dim,
        decoder_dim=args.decoder_dim,
        vocab_size=len(tokenizer),
        device=DEVICE,
        dropout=args.dropout,
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
    scheduler = CosineAnnealingLR(optimizer, T_max=8, eta_min=1e-05)

    for epoch in range(args.num_epochs):
        train_loop(args, train_loader, encoder, decoder, criterion, optimizer,
                   epoch, scheduler)
        val_loop(args, val_loader, encoder, decoder, criterion, tokenizer,
                 max_seq_length)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='/workdir/data/experiments/test/',
                        help='path for saving trained models')
    parser.add_argument('--log_step', type=int, default=500,
                        help='step size for prining log info')
    parser.add_argument('--encoder_pretrain', type=str, default='',
                        help='encoder pretrain path')
    parser.add_argument('--decoder_pretrain', type=str, default='',
                        help='decoder pretrain path')
    parser.add_argument('--output_height', type=int, default=150,
                        help='Height of images in dataset')
    parser.add_argument('--output_width', type=int, default=300,
                        help='Max width of images in dataset')

    # Model parameters
    parser.add_argument('--attention_dim', type=int, default=256,
                        help='size of the attention network')
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='input size of embedding network')
    parser.add_argument('--decoder_dim', type=int, default=512,
                        help='input size of decoder network')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_dataset_len', type=int, default=300000)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    main(args)
