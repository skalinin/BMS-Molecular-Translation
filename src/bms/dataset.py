import torch
from torch.utils.data import Dataset
import pandas as pd
import random
import cv2


def collate_fn(data):
    """
    Get from https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning

    Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, text = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    lengths = torch.LongTensor(lengths)
    return images, targets, lengths, text


class BMSDataset(Dataset):
    def __init__(self, data_csv, restrict_dataset_len=None, transform=None):
        super().__init__()
        self.transform = transform
        self.data_csv_len = data_csv.shape[0] - 1
        self.restrict_dataset_len = restrict_dataset_len
        self.image_paths = data_csv['image_path'].values
        self.inchi_text = data_csv['InChI_text'].values
        self.inchi_tokens = data_csv['InChI_index'].values

    def __len__(self):
        if self.restrict_dataset_len is not None:
            return self.restrict_dataset_len
        return self.data_csv_len

    def __getitem__(self, idx):
        if self.restrict_dataset_len is not None:
            idx = random.randint(0, self.data_csv_len)
        image_path = self.image_paths[idx]
        target = self.inchi_tokens[idx]
        text = self.inchi_text[idx]
        image = cv2.imread(image_path)
        if self.transform is not None:
            image = self.transform(image)

        target = torch.Tensor(target)
        return image, target, text
