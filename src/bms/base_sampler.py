from torch.utils.data import Sampler
import numpy as np
from collections import Counter


class BaseSampler(Sampler):
    """Make sequence of dataset indexes for batch sampler.

    Args:
        dataset (torch.utils.data.Dataset): Torch dataset or ConcatDataset
        folder2freq (dict): Batchfolder frequence parameters.
        batchfolder_func (dict): Dictionaty with batchfolders and its
            corresponding functions.
        hard_normalization (bool, optional): Normalize sample2prob to sum to 1.
            May not to sum to 1 if probs are too small numbers.
        dataset_len (int, optional): Length of output dataset (by default it
            is equal to the length of the input dataset).
    """
    def __init__(
        self, dataset, folder2freq, batchfolder_func, dataset_len=None,
        hard_normalization=True
    ):
        self.dataset = dataset
        self.hard_normalization = hard_normalization
        self.folder2freq = folder2freq
        self.batchfolder_func = batchfolder_func
        if dataset_len is not None:
            self.dataset_len = dataset_len
        else:
            self.dataset_len = len(self.dataset)
        self.sample2prob = self._sample2prob()

    def _apply_batchfolder_func(self, *args):
        for folder, freq in self.folder2freq.items():
            if self.batchfolder_func[folder](folder, *args):
                return folder
        raise Exception('All samples in dataset must be in batchfolder')

    def _sample2folder(self):
        """Define folder-name for each sample in dataset."""
        sample2folder = []
        for sample in self.dataset:
            sample2folder.append(self._apply_batchfolder_func(*sample))
        return sample2folder

    def _folder2sample_prob(self, sample2folder):
        """Define probability for a single sample in each folder
        to be added in batch.
        """
        folder2samples_count = Counter(sample2folder)
        total_folder_freq = sum(x for x in self.folder2freq.values())
        folder2sample_prob = {}
        for folder, samples_count in folder2samples_count.items():
            folder_prob = self.folder2freq[folder] / total_folder_freq
            sample_prob = folder_prob / samples_count
            folder2sample_prob[folder] = sample_prob
        return folder2sample_prob

    def _hard_normalization(self, sample2prob):
        """Probabilities normalization to make them sum to 1."""
        sample2prob /= sample2prob.sum()
        return sample2prob

    def _sample2prob(self):
        """Make list of samples' probabilities to be added in batch.
        The length of the list is equal to the length of the dataset.
        """
        sample2folder = self._sample2folder()
        folder2sample_prob = self._folder2sample_prob(sample2folder)
        sample2prob = np.array(
            [folder2sample_prob[folder] for folder in sample2folder])
        if self.hard_normalization:
            sample2prob = self._hard_normalization(sample2prob)
        return sample2prob

    def __iter__(self):
        dataset_indexes = np.random.choice(
            len(self.dataset), self.dataset_len, p=self.sample2prob)
        return iter(dataset_indexes)

    def __len__(self):
        return self.dataset_len
