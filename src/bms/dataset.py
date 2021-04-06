import torch
from torch.utils.data import Dataset
import random
import cv2

from bms.base_sampler import BaseSampler


def medium_sequence_sampler(folder_name, sample_len):
    if sample_len < 170:
        return True
    return False


def long_sequence_sampler(folder_name, sample_len):
    if sample_len >= 170:
        return True
    return False


BATCHFOLDER_FUNC = {
    "medium_sequence": medium_sequence_sampler,
    "long_sequence": long_sequence_sampler,
}


class SequentialSampler(BaseSampler):
    def __init__(self, dataset, folder2freq, dataset_len):
        super().__init__(dataset, folder2freq, BATCHFOLDER_FUNC, dataset_len)

    def _sample2folder(self):
        """Define folder-name for each sample in dataset."""
        samples_lengths = self.dataset['InChI_index_len'].values
        sample2folder = []
        for sample_len in samples_lengths:
            sample2folder.append(
                self._apply_batchfolder_func(sample_len)
            )
        return sample2folder


def collate_fn(data):
    """
    Get from https://github.com/yunjey/pytorch-tutorial/tree/master/
        tutorials/03-advanced/image_captioning

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
    def __init__(self, data_csv, transform=None):
        super().__init__()
        self.transform = transform
        self.data_csv_len = len(data_csv)
        self.image_paths = data_csv['image_path'].values
        self.inchi_text = data_csv['InChI_text'].values
        self.inchi_tokens = data_csv['InChI_index'].values

    def __len__(self):
        return self.data_csv_len

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        target = self.inchi_tokens[idx]
        text = self.inchi_text[idx]
        image = cv2.imread(image_path)
        if self.transform is not None:
            image = self.transform(image)

        target = torch.Tensor(target)
        return image, target, text


class BMSSumbissionDataset(Dataset):
    def __init__(self, data_csv, transform=None):
        super().__init__()
        self.transform = transform
        self.data_csv = data_csv
        self.image_paths = data_csv['image_path'].values

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image
