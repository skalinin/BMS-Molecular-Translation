import os
import pandas as pd
from tqdm import tqdm

from bms.utils import make_dir, generate_synth_images

tqdm.pandas()


IMAGES_PATH = '/workdir/data/TOX21/images/'
INIT_TXT_DATASETS_PATH = '/workdir/data/TOX21/txt_datasets/'
TRAIN_CSV_PATH = '/workdir/data/TOX21/train_processed.csv'
NUM_PRCESS = 8


def get_tox_list_smiles(tox_txt_path):
    tox_txt_dataset_paths = []
    files = os.listdir(tox_txt_path)
    for file in files:
        file_path = os.path.join(tox_txt_path, file)
        if os.path.isfile(file_path):
            tox_txt_dataset_paths.append(file_path)

    smiles_set = set()
    for tox_txt_dataset_path in tox_txt_dataset_paths:
        with open(tox_txt_dataset_path, 'r') as f:
            smiles = f.read().splitlines()
            for smile in smiles:
                smile = smile[:-1].strip()
                smiles_set.add(smile)
    return list(smiles_set)


def preprocess_data():
    """Preprocess TOX21 csv and generate synth images.

    TOX21 targets from http://www.dna.bio.keio.ac.jp/smiles/
    """

    make_dir(IMAGES_PATH)

    tox_list_smiles = get_tox_list_smiles(INIT_TXT_DATASETS_PATH)

    data_csv = pd.DataFrame(tox_list_smiles, columns=['Smile'])
    data_csv['index_col'] = data_csv.index
    data_csv['image_path'] = data_csv.progress_apply(
        lambda x: os.path.join(IMAGES_PATH, f"train_{x['index_col']}.png"),
        axis=1)

    generate_synth_images(
        data_csv['Smile'].values,
        data_csv['image_path'].values,
        NUM_PRCESS
    )

    data_csv.to_csv(TRAIN_CSV_PATH)


if __name__ == '__main__':
    preprocess_data()
