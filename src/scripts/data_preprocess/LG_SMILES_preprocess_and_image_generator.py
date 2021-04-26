import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

from bms.utils import noisy_smile

tqdm.pandas()


def preprocess_data():
    # smiles targets from https://www.kaggle.com/cpmpml/lg-smiles-solutions
    save_path = '/workdir/data/LG_SMILES/images/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    lg_smiles_csv = pd.read_csv('/workdir/data/LG_SMILES/train.csv')
    lg_smiles_csv = lg_smiles_csv.rename(columns={'SMILES': 'Smile'})
    lg_smiles_csv['image_path'] = lg_smiles_csv.progress_apply(
        lambda x: os.path.join(save_path, x['file_name']), axis=1)

    with Pool(8) as p:
        p.starmap(
            noisy_smile,
            zip(lg_smiles_csv['Smile'].values,
                lg_smiles_csv['image_path'].values)
        )

    lg_smiles_csv.to_csv('/workdir/data/LG_SMILES/train_processed.csv')


if __name__ == '__main__':
    preprocess_data()
