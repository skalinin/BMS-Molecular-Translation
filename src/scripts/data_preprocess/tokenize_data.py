import pandas as pd
from tqdm import tqdm
import torch

from bms.tokenizer import atomwise_tokenizer, Tokenizer

tqdm.pandas()


TRAIN_CSV_PATHS = [
    '/workdir/data/bms-molecular-translation/train_labels_processd.csv',
    # '/workdir/data/LG_SMILES/train_processed.csv',
    # '/workdir/data/TOX21/train_processed.csv'
]

VAL_CSV_PATH = '/workdir/data/bms-molecular-translation/val_labels_processd.csv'


def tokenize_data():
    # concat train and val lists of csv paths
    csv_paths = TRAIN_CSV_PATHS + [VAL_CSV_PATH]
    csv_len = []

    # create one data_csv to tokenize it later
    data_csv = pd.DataFrame()
    for csv_path in csv_paths:
        csv = pd.read_csv(csv_path)
        csv_len.append(len(csv))
        data_csv = pd.concat([csv[['image_path', 'Smile']], data_csv],
                             ignore_index=True)

    # create tokens from smile targets
    data_csv['Tokens'] = data_csv['Smile'].progress_apply(atomwise_tokenizer)

    # create tokenizer vocab
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data_csv['Tokens'].values)
    print(tokenizer.token2idx)
    torch.save(tokenizer, '/workdir/data/processed/tokenizer.pth')

    # create index for each sample label
    data_csv['Tokens_indexes'] = \
        data_csv['Tokens'].progress_apply(tokenizer.text_to_sequence)
    data_csv['Tokens_len'] = \
        data_csv['Tokens_indexes'].progress_apply(len)

    # split on train and val again
    train_csv = data_csv.iloc[:len(data_csv)-csv_len[-1], :].copy(deep=True)
    val_csv = data_csv.iloc[len(data_csv)-csv_len[-1]:, :].copy(deep=True)
    print(train_csv.iloc[0])
    print("len train csv", len(train_csv))
    print("len val csv", len(val_csv))

    # to properly save lists save as pickle
    train_csv.to_pickle('/workdir/data/processed/train_labels_processed.pkl')
    val_csv.to_pickle('/workdir/data/processed/val_labels_processed.pkl')


if __name__ == '__main__':
    tokenize_data()
