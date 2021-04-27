import pandas as pd
from tqdm import tqdm
import torch
from collections import Counter

from bms.tokenizer import atomwise_tokenizer, Tokenizer
from bms.model_config import model_config

tqdm.pandas()


TRAIN_CSV_PATH = '/workdir/data/bms-molecular-translation/train_labels_processd.csv'
VAL_CSV_PATH = '/workdir/data/bms-molecular-translation/val_labels_processd.csv'
EXTERNAL_TRAIN_CSV_PATHS = [
    '/workdir/data/LG_SMILES/train_processed.csv',
    '/workdir/data/TOX21/train_processed.csv'
]


def is_remove_row(tokens, tokens_to_remove: set):
    if tokens_to_remove.intersection(set(tokens)):
        return True
    return False


def remove_rare_tokens(data_csv, token_count_to_accept=0, save_tokens=[]):
    if token_count_to_accept > 0:
        token2counts = \
            Counter(c for clist in data_csv['Tokens'].values for c in clist)
        tokens_to_remove = set()
        for token, counts in token2counts.items():
            if (
                counts < token_count_to_accept
                and token not in save_tokens
            ):
                tokens_to_remove.add(token)
        to_romove = data_csv['Tokens'].progress_apply(
            is_remove_row, tokens_to_remove=tokens_to_remove)
        data_csv = data_csv.drop(data_csv[to_romove].index)
    return data_csv


def split_to_tokens(data_csv, tokenizer, token_count_to_accept=0):
    data_csv['Tokens'] = data_csv['Smile'].progress_apply(atomwise_tokenizer)
    data_csv = remove_rare_tokens(
        data_csv,
        token_count_to_accept,
        tokenizer.vocab  # do not remove tokens that alredy in tokenizer
    )
    tokenizer.add_texts(data_csv['Tokens'].values)
    return data_csv


def load_csv_data(train_csv_path, val_csv_path, external_train_csv_paths):
    train_csv = pd.read_csv(train_csv_path)
    val_csv = pd.read_csv(val_csv_path)
    external_data_csv = pd.DataFrame()
    for external_train_csv_path in EXTERNAL_TRAIN_CSV_PATHS:
        data_csv = pd.read_csv(external_train_csv_path)
        external_data_csv = pd.concat(
            [data_csv[['image_path', 'Smile']], external_data_csv],
            ignore_index=True
        )
    return train_csv, val_csv, external_data_csv


def tokens_to_indexes(data_csv, tokenizer):
    data_csv['Tokens_indexes'] = \
        data_csv['Tokens'].progress_apply(tokenizer.text_to_sequence)
    data_csv['Tokens_len'] = data_csv['Tokens_indexes'].apply(len)
    return data_csv


def tokenize_data():
    tokenizer = Tokenizer()

    train_csv, val_csv, external_data_csv = load_csv_data(
        TRAIN_CSV_PATH, VAL_CSV_PATH, EXTERNAL_TRAIN_CSV_PATHS)

    train_csv = split_to_tokens(train_csv, tokenizer)
    val_csv = split_to_tokens(val_csv, tokenizer)
    external_data_csv = split_to_tokens(external_data_csv, tokenizer, 300)

    tokenizer.fit_on_texts()
    print(tokenizer.token2idx)
    torch.save(tokenizer, model_config["paths"]["tokenizer"])

    train_csv = tokens_to_indexes(train_csv, tokenizer)
    val_csv = tokens_to_indexes(val_csv, tokenizer)
    external_data_csv = tokens_to_indexes(external_data_csv, tokenizer)

    train_csv = pd.concat(
        [
            train_csv[['image_path', 'Smile', 'Tokens_indexes', 'Tokens_len']],
            external_data_csv[['image_path', 'Smile', 'Tokens_indexes', 'Tokens_len']]
        ],
        ignore_index=True
    )

    # to properly save lists save as pickle
    train_csv.to_pickle(model_config["paths"]["train_csv"])
    val_csv.to_pickle(model_config["paths"]["val_csv"])


if __name__ == '__main__':
    tokenize_data()
