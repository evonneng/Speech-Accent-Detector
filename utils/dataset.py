"""
Module that contains that some utility functions
"""

import os
import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset
from .setup import *

class ActionDataset(Dataset):
    def __init__(self, data_dir, language, crop=None):
        self.data_dir = data_dir
        self.crop = crop

        self.trajs = os.listdir(data_dir)

        self._cache = {}

        self.language = language

    def __len__(self):
        return int(len(self.trajs) * 0.7)

    def __getitem__(self, idx):
        idx = idx + 1
        if idx not in self._cache:
            wav = get_wav(os.path.join(self.data_dir, self.language + str(idx) + '.wav'))
            mfcc = np.transpose(to_mfcc(wav))
            self._cache[idx] = mfcc

        mfcc = self._cache[idx]
        if self.crop is not None:
            s = np.random.choice(mfcc.shape[0] - self.crop + 1)
            mfcc = mfcc[s:s + self.crop, :]
        return mfcc


def load(data_filepath, language, num_workers=0, batch_size=32, **kwargs):
    dataset = ActionDataset(data_filepath, language, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)