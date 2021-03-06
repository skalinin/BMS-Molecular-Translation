import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import torch
import argparse
from rdkit import Chem

from bms.utils import get_file_path
from bms.dataset import BMSSumbissionDataset
from bms.transforms import get_val_transforms
from bms.model import EncoderCNN, DecoderWithAttention
from bms.model_config import model_config
from bms.utils import load_pretrain_model

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

tqdm.pandas()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_inchi_from_smile(smile):
    inchi = 'InChI=1S/'
    try:
        inchi = Chem.MolToInchi(Chem.MolFromSmiles(smile))
    except:
        pass
    return inchi


def test_loop(data_loader, encoder, decoder, tokenizer, max_seq_length):
    if decoder.training:
        decoder.eval()
    if encoder.training:
        encoder.eval()

    text_preds = []
    tq = tqdm(data_loader, total=len(data_loader))
    for images in tq:
        images = images.to(DEVICE)
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                features = encoder(images)
                predictions = decoder.predict(
                    features, max_seq_length, tokenizer.token2idx["<sos>"])
        predicted_sequence = torch.argmax(predictions.detach().cpu(), -1).numpy()
        text_preds.append(
            tokenizer.predict_captions(predicted_sequence))
    return np.concatenate(text_preds)


def main(args):
    test_csv = pd.read_csv(model_config["paths"]["submission_csv"])
    test_csv['image_path'] = test_csv['image_id'].progress_apply(
        get_file_path, main_folder='test')

    tokenizer = torch.load(model_config["paths"]["tokenizer"])

    test_transform = get_val_transforms(
        model_config['image_height'], model_config['image_width'])
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

    predictions = test_loop(test_loader, encoder, decoder, tokenizer,
                            args.max_seq_length)

    test_csv['Smile'] = predictions
    test_csv['InChI'] = test_csv['Smile'].progress_apply(make_inchi_from_smile)
    test_csv[['image_id', 'InChI']].to_csv(
        os.path.join(args.submission_path, 'submission.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission_path', type=str,
                        default='/workdir/data/experiments/',
                        help='path for saving submission csv')
    parser.add_argument('--encoder_pretrain', type=str, default='',
                        help='encoder pretrain path')
    parser.add_argument('--decoder_pretrain', type=str, default='',
                        help='decoder pretrain path')
    parser.add_argument('--max_seq_length', type=int, default=110,
                        help='max sequenxe lenght to decode')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    main(args)
