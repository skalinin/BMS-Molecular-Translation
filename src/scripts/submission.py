import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import torch
import argparse

from bms.utils import get_file_path
from bms.dataset import BMSSumbissionDataset
from bms.transforms import get_val_transforms
from bms.model import EncoderCNN, DecoderWithAttention
from bms.utils import load_pretrain_model

tqdm.pandas()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_loop(args, data_loader, encoder, decoder, tokenizer, max_seq_length):
    if decoder.training:
        decoder.eval()
    if encoder.training:
        encoder.eval()

    text_preds = []
    tq = tqdm(data_loader, total=len(data_loader))
    for images in tq:
        images = images.to(DEVICE)
        with torch.no_grad():
            features = encoder(images)
            predictions = decoder.predict(features, max_seq_length, tokenizer)
        predicted_sequence = torch.argmax(predictions.detach().cpu(), -1).numpy()
        text_preds.append(
            tokenizer.predict_captions(predicted_sequence))
    return np.concatenate(text_preds)


def main(args):
    data_csv = pd.read_pickle(
        '/workdir/data/processed/train_labels_processed.pkl')
    max_seq_length = data_csv['InChI_index'].map(len).max()

    test_csv = pd.read_csv(
        '/workdir/data/bms-molecular-translation/sample_submission.csv')
    test_csv['image_path'] = test_csv['image_id'].progress_apply(
        get_file_path, main_folder='test')

    tokenizer = torch.load('/workdir/data/processed/tokenizer.pth')

    test_transform = get_val_transforms(args.output_height, args.output_width)
    test_dataset = BMSSumbissionDataset(
        data_csv=test_csv,
        transform=test_transform
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

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

    predictions = test_loop(args, test_loader, encoder, decoder, tokenizer, max_seq_length)

    test_csv['InChI'] = [f"InChI=1S/{text}" for text in predictions]
    test_csv[['image_id', 'InChI']].to_csv(
        os.path.join(args.submission_path, 'submission.csv'), index=False)
    print(test_csv.head())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument('--attention_dim', type=int, default=256,
                        help='size of the attention network')
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='input size of embedding network')
    parser.add_argument('--decoder_dim', type=int, default=512,
                        help='input size of decoder network')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--output_height', type=int, default=224,
                        help='Height of images in dataset')
    parser.add_argument('--output_width', type=int, default=224,
                        help='Max width of images in dataset')

    parser.add_argument('--submission_path', type=str,
                        default='/workdir/data/experiments/',
                        help='path for saving submission csv')
    parser.add_argument('--encoder_pretrain', type=str, default='',
                        help='encoder pretrain path')
    parser.add_argument('--decoder_pretrain', type=str, default='',
                        help='decoder pretrain path')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    main(args)
