import torch
import torch.nn as nn


class Songs(torch.utils.data.Dataset):

    def __init__(self):
        super().__init__()
        self.len = 10000

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        sample = {}

        return sample
