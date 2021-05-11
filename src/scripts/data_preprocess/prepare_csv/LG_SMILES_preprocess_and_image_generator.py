import os
import pandas as pd
from tqdm import tqdm

from bms.utils import make_dir, generate_synth_images

tqdm.pandas()


IMAGES_PATH = '/workdir/data/LG_SMILES/images/'
INIT_CSV_PATH = '/workdir/data/LG_SMILES/train.csv'
TRAIN_CSV_PATH = '/workdir/data/LG_SMILES/train_processed.csv'
NUM_PRCESS = 8


def preprocess_data():
    """Preprocess LG SMILES csv and generate synth images.

    LG SMILES targets from https://www.kaggle.com/cpmpml/lg-smiles-solutions
    """

    make_dir(IMAGES_PATH)

    data_csv = pd.read_csv(INIT_CSV_PATH)
    data_csv = data_csv.rename(columns={'SMILES': 'Smile'})
    data_csv['image_path'] = data_csv.progress_apply(
        lambda x: os.path.join(IMAGES_PATH, x['file_name']), axis=1)

    generate_synth_images(
        data_csv['Smile'].values,
        data_csv['image_path'].values,
        NUM_PRCESS
    )

    data_csv.to_csv(TRAIN_CSV_PATH)


if __name__ == '__main__':
    preprocess_data()
