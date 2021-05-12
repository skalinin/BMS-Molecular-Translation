import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
import random
import cv2


def build_batches(sentences, batch_size, num_chunks_in_batch=1):
    """
    Randomize batches sequences along sentences if dataset indexes.
    https://gist.github.com/pommedeterresautee/1a334b665710bec9bb65965f662c94c8#file-trainer-py-L181
    https://wandb.ai/pommedeterresautee/speed_training/reports/Train-HuggingFace-Models-Twice-As-Fast--VmlldzoxMDgzOTI

    Args:
        sentences (list): List of samples.
        batch_size (int): Batch size.
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


class SequentialSampler(Sampler):
    """Make sequence of dataset indexes for batch sampler.

    Args:
        dataset (torch.utils.data.Dataset): Torch dataset or ConcatDataset
        dataset_len (int, optional): Length of output dataset (by default it
            is equal to the length of the input dataset).
        batch_size (int, optional): Batch size, only used in smartbatching.
        smart_batching (bool, optional): Apply smartbatching, default is False.
        init_sample_probs (list, optional): list of samples' probabilities to
            be added in batch. If None probs for all samples would be the same.
            The length of the list must be equal to the length of the dataset.
    """
    def __init__(
        self, dataset, dataset_len=None, batch_size=None, smart_batching=False,
        init_sample_probs=None
    ):
        self.dataset = dataset
        if dataset_len is not None:
            self.dataset_len = dataset_len
        else:
            self.dataset_len = len(self.dataset)
        self.batch_size = batch_size
        self.smart_batching = smart_batching
        if init_sample_probs is None:
            self.init_sample_probs = \
                np.array([1. for i in range(len(self.dataset))],
                         dtype=np.float64)
        else:
            self.init_sample_probs = np.array(init_sample_probs,
                                              dtype=np.float64)
            assert len(self.init_sample_probs) == len(self.dataset), "The len \
                of the sample_probs must be equal to the len of the dataset."

    def smart_batches(self, dataset_indexes):
        """Sort inexex by samples length to make LSTM training faster.
        May affect the quality of the training.
        """
        samples_len = \
            self.dataset.iloc[dataset_indexes]['Tokens_len'].values
        sorted_indexes = [
            idx for _, idx in
            sorted(zip(samples_len, dataset_indexes), reverse=True)
        ]
        batched_sorted_indexes = build_batches(sorted_indexes, self.batch_size)
        return batched_sorted_indexes

    def _sample_probs_normalization(self):
        """Probabilities normalization to make them sum to 1.
        Sum might not be equal to 1 if probs are too small.
        """
        return self.init_sample_probs / self.init_sample_probs.sum()

    def update_sample_probs(self, probs, idxs, k):
        """Update probabilities of samples to be added in batch on the
        next epoch.
        """
        for prob, idx in zip(probs, idxs):
            self.init_sample_probs[idx] = prob ** k

    def __iter__(self):
        sample_probs = self._sample_probs_normalization()
        dataset_indexes = np.random.choice(
            len(self.dataset), self.dataset_len, p=sample_probs)
        if self.smart_batching:
            dataset_indexes = self.smart_batches(dataset_indexes)
        return iter(dataset_indexes)

    def __len__(self):
        return self.dataset_len


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
    images, captions, lengths, text, idxs = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    lengths = torch.LongTensor(lengths)
    return images, targets, lengths, text, idxs


class BMSDataset(Dataset):
    def __init__(self, data_csv, transform=None):
        super().__init__()
        self.transform = transform
        self.data_csv_len = len(data_csv)
        self.image_paths = data_csv['image_path'].values
        self.inchi_text = data_csv['Smile'].values
        self.inchi_tokens = data_csv['Tokens_indexes'].values
        self.inchi_lengths = data_csv['Tokens_len'].values

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
        return image, target, length, text, idx


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
