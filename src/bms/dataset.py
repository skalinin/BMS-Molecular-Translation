import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2


class BMSDataset(Dataset):
    def __init__(self, data_pickle_path, transform=None):
        super().__init__()
        self.transform = transform
        data_csv = pd.read_pickle(data_pickle_path)
        self.image_paths = data_csv['image_path'].values
        self.inchi_tokens = data_csv['InChI_index'].values

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        target = self.inchi_tokens[idx]
        target_length = len(target)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.transform is not None:
            image = self.transform(image)
        return image, target, target_length
