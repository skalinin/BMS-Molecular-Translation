import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

from bms.utils import noisy_smile

tqdm.pandas()


def get_tox_txt_datasets(tox_txt_path):
    tox_txt_datasets = []
    files = os.listdir(tox_txt_path)
    for file in files:
        file_path = os.path.join(tox_txt_path, file)
        if os.path.isfile(file_path):
            tox_txt_datasets.append(file_path)
    return tox_txt_datasets


def preprocess_data():
    # smiles targets from http://www.dna.bio.keio.ac.jp/smiles/
    save_path = '/workdir/data/TOX21/images/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    tox_txt_dataset_paths = get_tox_txt_datasets('/workdir/data/TOX21/txt_datasets/')

    smiles_set = set()
    for tox_txt_dataset_path in tox_txt_dataset_paths:
        with open(tox_txt_dataset_path, 'r') as f:
            smiles = f.read().splitlines() 
            for smile in smiles:
                smile = smile[:-1].strip()
                smiles_set.add(smile)

    data_csv = pd.DataFrame(list(smiles_set), columns =['Smile'])
    data_csv['index_col'] = data_csv.index
    data_csv['image_path'] = data_csv.progress_apply(
        lambda x: os.path.join(save_path, f"train_{x['index_col']}.png"), axis=1)

    with Pool(8) as p:
        p.starmap(
            noisy_smile,
            zip(data_csv['Smile'].values,
                data_csv['image_path'].values)
        )

    data_csv.to_csv('/workdir/data/TOX21/train_processed.csv')


if __name__ == '__main__':
    preprocess_data()
