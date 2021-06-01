import os
import pandas as pd
from tqdm import tqdm

from bms.tokenizer import inchi2smile
from bms.utils import make_dir, generate_synth_images

tqdm.pandas()


IMAGES_PATH = '/workdir/data/extra_approved_InChIs/'
INIT_CSV_PATH = '/workdir/data/bms-molecular-translation/extra_approved_InChIs.csv'
TRAIN_CSV_PATH = '/workdir/data/extra_approved_InChIs/extra_approved_InChIs_processed.csv'
NUM_IMAGES = 1500000
DROP_SAMPLES = [484865]  # rdkit stuck on this inchis
NUM_PRCESS = 8


def preprocess_data():
    """Preprocess extra_approved_InChIs.csv and generate synth images."""

    make_dir(IMAGES_PATH)

    data_csv = pd.read_csv(INIT_CSV_PATH)
    data_csv = data_csv.drop(DROP_SAMPLES)

    data_csv = data_csv.iloc[:NUM_IMAGES].copy(deep=True)

    data_csv['Smile'] = data_csv['InChI'].progress_apply(inchi2smile)
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
