import torch
from torch.utils.data import Dataset
import numpy as np
import random
import cv2

from bms.base_sampler import BaseSampler


def build_batches(sentences, batch_size, num_chunks_in_batch=1):
    """
    Randomize batches sequences along sentences if dataset indexes.
    https://gist.github.com/pommedeterresautee/1a334b665710bec9bb65965f662c94c8#file-trainer-py-L181
    https://wandb.ai/pommedeterresautee/speed_training/reports/Train-HuggingFace-Models-Twice-As-Fast--VmlldzoxMDgzOTI

    Args:
        sentences (list): List of samples.
        batch_size (int): Batch size,
        num_chunks_in_batch (int, optional): Training batch should consist of
            several "chunks" of samples with different sequece lengths to make
            training more random. Default is 1, which mean in batch would be
            samples with the same length.
    """
    chunk_size = int(batch_size/num_chunks_in_batch)
    batch_ordered_sentences = list()
    while len(sentences) > 0:
        to_take = min(chunk_size, len(sentences))
        select = random.randint(0, len(sentences) - to_take)
        batch_ordered_sentences += sentences[select:select + to_take]
        del sentences[select:select + to_take]
    return batch_ordered_sentences


class SequentialSampler(BaseSampler):
    def __init__(self, dataset, folder2freq, dataset_len, batch_size):
        self.batch_size = batch_size
        super().__init__(dataset, folder2freq, None, dataset_len)

    def _sample2folder(self):
        """Define folder-name for each sample in dataset.
        Sample lenght is used as folders.
        """
        return self.dataset['Smile_index_len'].values

    def smart_batches(self, dataset_indexes):
        """Sort inexex by samples length to make LSTM training faster."""
        samples_len = \
            self.dataset.iloc[dataset_indexes]['Smile_index_len'].values
        sorted_indexes = [
            idx for _, idx in
                sorted(zip(samples_len, dataset_indexes), reverse=True)
        ]
        batched_sorted_indexes = build_batches(sorted_indexes, self.batch_size)
        return batched_sorted_indexes

    def __iter__(self):
        dataset_indexes = np.random.choice(
            len(self.dataset), self.dataset_len, p=self.sample2prob)
        # dataset_indexes = self.smart_batches(dataset_indexes)
        return iter(dataset_indexes)


def collate_fn(data):
    """
    Get from https://github.com/yunjey/pytorch-tutorial/tree/master/
        tutorials/03-advanced/image_captioning

    Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, lengths, text = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
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
        self.inchi_text = data_csv['Smile'].values
        self.inchi_tokens = data_csv['Smile_index'].values
        self.inchi_lengths = data_csv['Smile_index_len'].values

    def __len__(self):
        return self.data_csv_len

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        target = self.inchi_tokens[idx]
        text = self.inchi_text[idx]
        length = self.inchi_lengths[idx]
        image = cv2.imread(image_path)
        if self.transform is not None:
            image = self.transform(image)

        target = torch.Tensor(target)
        return image, target, length, text


class BMSSumbissionDataset(Dataset):
    def __init__(self, data_csv, transform=None):
        super().__init__()
        self.transform = transform
        self.data_csv_len = len(data_csv)
        self.image_paths = data_csv['image_path'].values

    def __len__(self):
        return self.data_csv_len

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image
