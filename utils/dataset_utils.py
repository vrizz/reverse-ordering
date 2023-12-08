"""
Some functions to create dataset for Pytorch training.
"""

import torch


def transform_to_tensor(x):
    return torch.from_numpy(x)
