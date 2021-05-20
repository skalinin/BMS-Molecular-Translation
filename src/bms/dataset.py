import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
import random
import cv2


def apply_smart_batching(epoch_sample_indexes, seq_lenghts, batch_size):
    """Sort indexes by samples length to make LSTM training faster.
    May affect the quality of the training.
    """
    epoch_seq_lengths = np.take(seq_lenghts, epoch_sample_indexes)
    sorted_indexes = [
        idx for _, idx in
        sorted(zip(epoch_seq_lengths, epoch_sample_indexes), reverse=True)
    ]
    batched_sorted_indexes = build_batches(sorted_indexes, batch_size)
    return batched_sorted_indexes


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
        dataset_len (int): Length of train dataset (by default it
            is equal to the length of the input dataset).
        epoch_size (int, optional): Size of train epoch (by default it
            is equal to the dataset_len). Can be specified if you need to
            reduce the time of the epoch.
        smart_batching (bool, optional): Apply smartbatching, default is False.
            To use smartbatching the batch_size and seq_lenghts must be specified.
        batch_size (int, optional): Batch size, only used in smartbatching.
        seq_lenghts (list, optional): List of sequences' lenghts, used in
            smartbatching to sort samples.
        init_sample_probs (list, optional): List of samples' probabilities to
            be added in batch. If None probs for all samples would be the same.
            The length of the list must be equal to the length of the dataset.
        sample_probs_power (int, optional): The degree to which sample probs
            is raised to make probs smoother/sharper. Default is 1 (no power).
    """
    def __init__(
        self, dataset_len, epoch_size=None, batch_size=None, seq_lenghts=None,
        smart_batching=False, init_sample_probs=None, sample_probs_power=1
    ):
        self.dataset_len = dataset_len
        if epoch_size is not None:
            self.epoch_size = epoch_size
        else:
            self.epoch_size = dataset_len

        # smartbatching params
        self.batch_size = batch_size
        self.smart_batching = smart_batching
        self.seq_lenghts = seq_lenghts
        if smart_batching:
            assert seq_lenghts is not None and batch_size is not None, \
                "Both seq_lenghts and batch_size muse be specified to use " \
                "smart batching."

        # sample probs params
        self.sample_probs_power = sample_probs_power
        if init_sample_probs is None:
            self.init_sample_probs = \
                np.array([1. for i in range(dataset_len)], dtype=np.float64)
        else:
            self.init_sample_probs = \
                np.array(init_sample_probs, dtype=np.float64)
            assert len(self.init_sample_probs) == dataset_len, "The len " \
                "of the sample_probs must be equal to the dataset_len."

    def _sample_probs_normalization(self, sample_probs):
        """Probabilities normalization to make them sum to 1.
        Sum might not be equal to 1 if probs are too small.
        """
        return sample_probs / sample_probs.sum()

    def _sample_probs_power(self):
        """Raise the init_sample_probs to the power of sample_probs_power to
        make probs smoother/sharper."""
        return self.init_sample_probs ** self.sample_probs_power

    def update_sample_probs(self, probs, idxs):
        """Update probabilities of samples to be added in batch on the
        next epoch."""
        for prob, idx in zip(probs, idxs):
            self.init_sample_probs[idx] = prob

    def __iter__(self):
        sample_probs = self._sample_probs_power()
        sample_probs = self._sample_probs_normalization(sample_probs)
        dataset_indexes = np.random.choice(
            a=self.dataset_len,
            size=self.epoch_size,
            p=sample_probs,
            replace=False,  # only unique samples inside an epoch
        )
        if self.smart_batching:
            dataset_indexes = apply_smart_batching(
                dataset_indexes, self.seq_lenghts, self.batch_size)
        return iter(dataset_indexes)

    def __len__(self):
        return self.epoch_size


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
