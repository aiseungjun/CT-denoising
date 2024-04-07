import os
import glob
from data import common
import numpy as np
import torch.utils.data as data
import torch
import math
import random
import pydicom
import mat73
from pathlib import Path
import time

from matplotlib import pyplot as plt

class SRData(data.Dataset):
    def __init__(self, config, mode='train', augment=False):
        self.mode = mode
        self.augment = augment
        self.device = torch.device('cpu' if config['cpu'] else 'cuda')
        self.ldct, self.ndct = self._scan()

    def __getitem__(self, idx):
        ldct, ndct = self._load_file(idx)
        ldct, ndct = self.preparation(ldct, ndct)
        # print(idx)
        # print(ldct.shape)
        # print(ndct.shape)
        return ldct.copy(), ndct.copy()

    def __len__(self):
        return self.ldct.shape[0]

    def _scan(self):
        patient_files = glob.glob(os.path.join(config.data_dir, self.mode))
        for i in patient_files:
            
        u_water = 0.0192
        # mm-1 to HU
        ldct = (ldct - u_water) * 1000 / u_water
        ndct = (ndct - u_water) * 1000 / u_water
        
        return ldct, ndct
        
    def _get_index(self, idx):
        return idx % self._length

    def _load_file(self, idx):
        idx = self._get_index(idx)
        
        ldct = self._ldct[:, :, idx].astype(np.float32)
        ndct = self._ndct[:, :, idx].astype(np.float32)

        return ldct, ndct

    def preparation(self, ldct, ndct):
        if self.mode == 'train' and self.augment:
            ldct, ndct = common.augment(ldct, ndct)
            # print('haha', np.expand_dims(ldct, 0).shape)
            # print('haha', np.expand_dims(ndct, 0).shape)
        ldct, ndct = common.get_patch(
            np.expand_dims(ldct, 0), np.expand_dims(ndct, 0)
            # np.expand_dims(ldct, 0), np.expand_dims(ndct, 0), 55
        )
        return ldct, ndct
