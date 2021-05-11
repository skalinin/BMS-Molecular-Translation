import pandas as pd
from tqdm import tqdm

from bms.utils import get_file_path
from bms.model_config import model_config
from bms.tokenizer import inchi2smile

tqdm.pandas()

INIT_CSV_PATH = '/workdir/data/bms-molecular-translation/train_labels.csv'
TRAIN_CSV_PATH = '/workdir/data/bms-molecular-translation/train_labels_processd.csv'
VAL_CSV_PATH = '/workdir/data/bms-molecular-translation/val_labels_processd.csv'


def preprocess_data():
    """Preprrocess kaggle BMS csv."""

    train_csv = pd.read_csv(INIT_CSV_PATH)

    train_csv['image_path'] = train_csv['image_id'].progress_apply(
        get_file_path, main_folder='train')
    train_csv['Smile'] = train_csv['InChI'].progress_apply(inchi2smile)

    train_data_size = int(model_config['train_dataset_size'] * len(train_csv))
    data_csv_train = train_csv.iloc[:train_data_size, :].copy(deep=True)
    data_csv_val = train_csv.iloc[train_data_size:, :].copy(deep=True)

    data_csv_train.to_csv(TRAIN_CSV_PATH)
    data_csv_val.to_csv(VAL_CSV_PATH)


if __name__ == '__main__':
    preprocess_data()
