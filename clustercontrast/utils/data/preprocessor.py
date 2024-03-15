from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image
import pickle


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        pid, type, view, fname = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        with open(fpath, 'rb') as f:
            img = pickle.load(f)

        return img, fname, pid, type, view, index


