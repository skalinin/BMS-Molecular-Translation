import torch
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn as nn
import pandas as pd
import os
import argparse
import time

from bms.dataset import collate_fn, BMSDataset
from bms.transforms import get_transforms
from bms.model import EncoderCNN, DecoderRNN
from bms.metrics import AverageMeter, time_remain, get_score

# from pudb import set_trace; set_trace()


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_loop(
    args, data_loader, encoder, decoder, criterion, optimizer, epoch
):
    total_step = len(data_loader)
    loss_avg = AverageMeter()
    start = time.time()
    if not decoder.training:
        decoder.train()
    if not encoder.training:
        encoder.train()

    for i, (images, captions, lengths, _) in enumerate(data_loader):
        # Set mini-batch dataset
        images = images.to(DEVICE)
        captions = captions.to(DEVICE)
        targets = \
            pack_padded_sequence(captions, lengths, batch_first=True)[0]
        # Forward, backward and optimize
        features = encoder(images)
        outputs = decoder(features, captions, lengths)
        loss = criterion(outputs, targets)
        loss_avg.update(loss.item(), args.batch_size)
        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()
        optimizer.step()
    # Print log info
    print('Epoch [{}/{}], Loss: {:.4f}'.format(
        epoch, args.num_epochs, loss_avg.avg))
    # Save the model checkpoints
    torch.save(decoder.state_dict(), os.path.join(
        args.model_path,
        'decoder-{}-{:.4f}.ckpt'.format(epoch+1, loss_avg.avg)))
    torch.save(encoder.state_dict(), os.path.join(
        args.model_path,
        'encoder-{}-{:.4f}.ckpt'.format(epoch+1, loss_avg.avg)))


def val_loop(args, data_loader, encoder, decoder, criterion, tokenizer):
    total_step = len(data_loader)
    loss_avg = AverageMeter()
    acc_avg = AverageMeter()
    start = time.time()
    if decoder.training:
        decoder.eval()
    if encoder.training:
        encoder.eval()

    for i, (images, captions, lengths, text_true) in enumerate(data_loader):
        # Set mini-batch dataset
        images = images.to(DEVICE)
        captions = captions.to(DEVICE)
        targets = \
            pack_padded_sequence(captions, lengths, batch_first=True)[0]
        # Forward
        with torch.no_grad():
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            sampled_ids = decoder.sample(features)
            sampled_ids = sampled_ids.cpu().numpy()
        text_preds = tokenizer.predict_captions(sampled_ids)
        acc_avg.update(get_score(text_true, text_preds), args.batch_size)
        loss_avg.update(loss.item(), args.batch_size)
    # Print log info
    print('Val step Loss: {:.4f}, Acc: {:.4f}'.format(
        loss_avg.avg, acc_avg.avg))


def get_loaders(args, data_csv):
    data_csv_len = data_csv.shape[0] - 1
    train_data_size = int(0.99*data_csv_len)
    data_csv_train = data_csv.iloc[:train_data_size,:]
    data_csv_val = data_csv.iloc[train_data_size:,:]

    train_transform = get_transforms((224, 224))
    val_transform = get_transforms((224, 224))
    train_dataset = BMSDataset(
        data_csv=data_csv_train,
        max_dataset_len=args.train_dataset_len,
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
    encoder = EncoderCNN(args.embed_size).to(DEVICE)
    decoder = DecoderRNN(
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        vocab_size=len(tokenizer.token2idx),
        num_layers=args.num_layers,
        max_seq_length=max_seq_length
    ).to(DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + \
        list(encoder.linear.parameters()) + \
        list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        # Train the models
        train_loop(
            args, train_loader, encoder, decoder, criterion, optimizer, epoch)
        val_loop(args, val_loader, encoder, decoder, criterion, tokenizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='/workdir/data/experiments/test/',
                        help='path for saving trained models')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_dataset_len', type=int, default=250000)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    main(args)
