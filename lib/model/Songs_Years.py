import torch
import torch.nn as nn


class Songs_Years(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(128, 128)

    def forward(self, input):
        x = self.layer(input)
        return x
