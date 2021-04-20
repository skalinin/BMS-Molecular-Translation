import pandas as pd
from tqdm import tqdm
import torch

from bms.utils import get_file_path
from bms.tokenizer import (
    split_InChI_to_tokens, Tokenizer, make_inchi_from_chem_dict,
    split_InChI_to_chem_groups
)
from bms.model_config import model_config

tqdm.pandas()


def tokenize_data():
    train_csv = pd.read_csv('/workdir/data/bms-molecular-translation/train_labels.csv')

    train_csv['image_path'] = train_csv['image_id'].progress_apply(
        get_file_path, main_folder='train')

    # preprocess InChI sample labels to tokens
    train_csv['InChI_chem_dict'] = train_csv['InChI'].progress_apply(
        split_InChI_to_chem_groups,
        chem_tokens=model_config['chem2start_token'].keys()
    )
    train_csv['InChI_text'] = train_csv['InChI_chem_dict'].progress_apply(
        make_inchi_from_chem_dict,
        chem_to_take=model_config['chem_predict'],
        skip_first_token=True
    )

    # remove unnecessary columns
    del train_csv['image_id']
    del train_csv['InChI_chem_dict']

    train_csv['InChI_tokens'] = \
        train_csv['InChI_text'].progress_apply(split_InChI_to_tokens)

    # create tokenizer vocab of InChI tokens
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_csv['InChI_tokens'].values)
    print(tokenizer.token2idx)
    torch.save(tokenizer, '/workdir/data/processed/tokenizer.pth')

    # create InChI index for each sample label
    train_csv['InChI_index'] = \
        train_csv['InChI_tokens'].progress_apply(tokenizer.text_to_sequence)
    train_csv['InChI_index_len'] = \
        train_csv['InChI_index'].progress_apply(len)

    # to properly save lists save as pickle
    print(train_csv.iloc[0])
    train_csv.to_pickle('/workdir/data/processed/train_labels_processed.pkl')


if __name__ == '__main__':
    tokenize_data()
