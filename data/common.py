import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import random


def get_patch(ldct, ndct, patch_size=64):
    # patch_size = 64
    ix = random.randint(0, 512 - patch_size)
    iy = random.randint(0, 512 - patch_size)
    ldct = ldct[:, ix:ix + patch_size, iy:iy + patch_size]
    ndct = ndct[:, ix:ix + patch_size, iy:iy + patch_size]
    
    return ldct, ndct

def augment(ldct, ndct, hflip=True, vflip=True, rot=True):    
    hflip = hflip and random.random() < 0.5
    vflip = vflip and random.random() < 0.5
    rot90 = rot * random.randint(0,4)

    # # write your code
    if hflip:
        ldct = ldct[:, :, ::-1]
        ndct = ndct[:, :, ::-1]
    
    if vflip:
        ldct = ldct[:, ::-1, :]
        ndct = ndct[:, ::-1, :]

    ldct = np.rot90(ldct, rot90, axes=(1,2))
    ndct = np.rot90(ndct, rot90, axes=(1,2))
    return ldct, ndct