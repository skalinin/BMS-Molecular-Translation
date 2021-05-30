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
    '/workdir/data/extra_approved_InChIs/extra_approved_InChIs_processed.csv',
]


def is_row_has_tokens_to_remove(tokens, tokens_to_remove: set):
    """Check if given tokens has no intersection with unnecessary tokens."""

    if tokens_to_remove.intersection(set(tokens)):
        return True
    return False


def is_too_long_row(tokens, seq_len_to_accept):
    """Check if the number of tokens does not exceed the allowed length."""

    if len(tokens) > seq_len_to_accept:
        return True
    return False


def remove_rare_tokens(data_csv, token_count_to_accept=0, save_tokens=[]):
    """Remove from data rows with rare tokens.

    Args:
        data_csv (pandas.core.frame.DataFrame): Pandas DataFrame.
        token_count_to_accept (int, optional): The min number of token
            occurrences in the data to leave rows with this token in the data.
            Default is 0, no rows will be dropped.
        save_token (list): List of tokens that should be preserved in the data
            despite of the number of their occurence.
    """

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
            is_row_has_tokens_to_remove, tokens_to_remove=tokens_to_remove)
        data_csv = data_csv.drop(data_csv[to_romove].index)
    return data_csv


def remove_long_sequences(data_csv, seq_len_to_accept=0):
    """Remove from data too long rows.

    Args:
        data_csv (pandas.core.frame.DataFrame): Pandas DataFrame.
        seq_len_to_accept (int, optional): The min len of sequence to be left
            in data. Default is 0, no rows will be dropped.
    """

    if seq_len_to_accept > 0:
        to_romove = data_csv['Tokens'].progress_apply(
            is_too_long_row, seq_len_to_accept=seq_len_to_accept)
        data_csv = data_csv.drop(data_csv[to_romove].index)
    return data_csv


def preprocess_data(data_csv, tokenizer, token_count_to_accept=0,
                    seq_len_to_accept=0):
    """
    Prepare data: split to tokens, remove rows with rare tokens or rows with
    too long sequences.

    Args:
        data_csv (pandas.core.frame.DataFrame): Pandas DataFrame.
        tokenizer (bms.tokenizer.Tokenizer): Tokenizer.
        token_count_to_accept (int, optional): The min number of token
            occurrences in the data to leave rows with this token in the data.
            Default is 0, no rows will be dropped.
        seq_len_to_accept (int, optional): The min len of sequence to be left
            in data. Default is 0, no rows will be dropped.
    """

    data_csv['Tokens'] = data_csv['Smile'].progress_apply(atomwise_tokenizer)
    data_csv = remove_rare_tokens(
        data_csv,
        token_count_to_accept,
        tokenizer.vocab  # do not remove tokens that alredy in tokenizer
    )
    data_csv = remove_long_sequences(data_csv, seq_len_to_accept)
    tokenizer.add_texts(data_csv['Tokens'].values)
    return data_csv


def load_csv_data(train_csv_path, val_csv_path, external_train_csv_paths):
    """Load train and val csv and also data from external sources.
    All external csv will be joined in one table.
    """

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
    """Convers tokens to indexes using bms.Tokenizer."""

    data_csv['Tokens_indexes'] = \
        data_csv['Tokens'].progress_apply(tokenizer.text_to_sequence)
    data_csv['Tokens_len'] = data_csv['Tokens_indexes'].apply(len)
    return data_csv


def tokenize_data():
    """Preprocess and tokenize data.

    Join train data and external data in one table.
    """

    tokenizer = Tokenizer()

    train_csv, val_csv, external_data_csv = load_csv_data(
        TRAIN_CSV_PATH, VAL_CSV_PATH, EXTERNAL_TRAIN_CSV_PATHS)

    train_csv = preprocess_data(train_csv, tokenizer)
    val_csv = preprocess_data(val_csv, tokenizer)
    external_data_csv = preprocess_data(external_data_csv, tokenizer, 1000, 100)

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

    print("Len train csv:", len(train_csv))
    print("Len val csv:", len(val_csv))

    # to properly save lists save as pickle
    train_csv.to_pickle(model_config["paths"]["train_csv"])
    val_csv.to_pickle(model_config["paths"]["val_csv"])


if __name__ == '__main__':
    tokenize_data()
