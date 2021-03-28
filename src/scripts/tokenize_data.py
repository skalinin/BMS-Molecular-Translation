import pandas as pd
from tqdm import tqdm
import torch

from bms.utils import get_file_path
from bms.tokenizer import split_InChI_to_tokens, Tokenizer

tqdm.pandas()


def tokenize_data():
    train_csv = pd.read_csv('/workdir/data/bms-molecular-translation/train_labels.csv')
    train_csv['image_path'] = train_csv['image_id'].apply(
        get_file_path, main_folder='train')

    # preprocess InChI sample labels to tokens
    train_csv['InChI_tokens'] = train_csv['InChI'].progress_apply(
        split_InChI_to_tokens)

    # create tokenizer vocab of InChI tokens
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_csv['InChI_tokens'].values)
    torch.save(tokenizer, '/workdir/data/processed/tokenizer.pth')

    # create InChI index for each sample label
    train_csv['InChI_index'] = train_csv['InChI_tokens'].progress_apply(
        tokenizer.text_to_sequence)

    # to properly save lists save as pickle
    train_csv.to_pickle('/workdir/data/processed/train_labels_processed.pkl')


if __name__ == '__main__':
    tokenize_data()
