import pandas as pd
from tqdm import tqdm
import torch

from bms.utils import get_file_path
from bms.tokenizer import (
    split_InChI_to_tokens, Tokenizer, make_inchi_from_chem_dict,
    split_InChI_to_chem_groups, get_chem_text_from_dict, text_to_sequence
)
from bms.model_config import model_config

tqdm.pandas()


def split_csv_to_train_val(data_csv):
    train_data_size = int(model_config['train_dataset_size'] * len(data_csv))
    train_csv = data_csv.iloc[:train_data_size, :].copy(deep=True)
    val_csv = data_csv.iloc[train_data_size:, :].copy(deep=True)
    return train_csv, val_csv


def preprocess_val_csv(val_csv):
    val_csv['image_path'] = val_csv['image_id'].progress_apply(
        get_file_path, main_folder='train')
    val_csv['InChI_chem_dict'] = val_csv['InChI'].progress_apply(
        split_InChI_to_chem_groups,
        chem_tokens=model_config['chem2start_token'].keys()
    )
    val_csv['InChI_chem_text'] = val_csv['InChI_chem_dict'].progress_apply(
        make_inchi_from_chem_dict, chem_to_take=model_config['chem_predict'])
    print(val_csv)
    val_csv.to_pickle('/workdir/data/processed/val_labels_processed.pkl')


def preprocess_train_csv(train_csv):
    train_csv['image_path'] = train_csv['image_id'].progress_apply(
        get_file_path, main_folder='train')

    # preprocess InChI sample labels to tokens
    train_csv['InChI_chem_dict'] = train_csv['InChI'].progress_apply(
        split_InChI_to_chem_groups,
        chem_tokens=model_config['chem2start_token'].keys()
    )
    chem_train_csv = pd.DataFrame()
    for chem_token in model_config['chem_predict']:
        df = train_csv.copy()
        df['InChI_chem_group'] = chem_token
        chem_train_csv = pd.concat([chem_train_csv, df], ignore_index=True)

    chem_train_csv = chem_train_csv.sort_values('image_id', ignore_index=True)

    chem_train_csv['InChI_chem_text'] = chem_train_csv.progress_apply(
        lambda x: get_chem_text_from_dict(
            chem_groups_dict=x['InChI_chem_dict'],
            key=x['InChI_chem_group']
        ),
        axis=1
    )

    # remove unnecessary columns
    del chem_train_csv['image_id']
    del chem_train_csv['InChI']
    del chem_train_csv['InChI_chem_dict']

    chem_train_csv['InChI_tokens'] = \
        chem_train_csv['InChI_chem_text'].progress_apply(split_InChI_to_tokens)

    # create tokenizer vocab of InChI tokens
    start_tokens = [model_config['chem2start_token'][chem]
                    for chem in model_config['chem_predict']]
    tokenizer = Tokenizer(start_tokens=start_tokens)
    tokenizer.fit_on_texts(chem_train_csv['InChI_tokens'].values)
    print(tokenizer.token2idx)
    torch.save(tokenizer, '/workdir/data/processed/tokenizer.pth')

    # create InChI index for each sample label
    chem_train_csv['InChI_index'] = chem_train_csv.progress_apply(
        lambda x: text_to_sequence(
            InChI_tokens=x['InChI_tokens'],
            tokenizer=tokenizer,
            InChI_chem_group=x['InChI_chem_group']
        ),
        axis=1
    )
    chem_train_csv['InChI_index_len'] = \
        chem_train_csv['InChI_index'].progress_apply(len)

    # to properly save lists save as pickle
    print(chem_train_csv)
    chem_train_csv.to_pickle('/workdir/data/processed/train_labels_processed.pkl')


def tokenize_data():
    data_csv = pd.read_csv('/workdir/data/bms-molecular-translation/train_labels.csv')

    train_csv, val_csv = split_csv_to_train_val(data_csv)

    preprocess_train_csv(train_csv)
    preprocess_val_csv(val_csv)


if __name__ == '__main__':
    tokenize_data()
