import torch
import torch.nn as nn
from lib.utils.logger import logger
from lib.utils.misc import param_size


class Songs_Years(nn.Module):

    def __init__(self, num_years=42, begin_year=1969):
        super().__init__()
        self.name = type(self).__name__
        self.begin_year = begin_year
        self.bn_avg = nn.BatchNorm1d(12)
        self.bn_cov = nn.BatchNorm1d(78)
        self.avg_encoder = nn.Sequential(nn.Linear(12, 128), nn.ReLU(), nn.Linear(128, 12))
        self.cov_encoder = nn.Sequential(nn.Linear(78, 128), nn.ReLU(), nn.Linear(128, 78))
        self.avg_encoder_layer = nn.Sequential(nn.Linear(12, 128), nn.ReLU(), nn.Linear(128, 128))
        self.cov_encoder_layer = nn.Sequential(nn.Linear(78, 128), nn.ReLU(), nn.Linear(128, 128))
        self.classification_layer = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, num_years))
        self.compute_loss = nn.CrossEntropyLoss()
        logger.info(f"{self.name} has {param_size(self)}M parameters")

    def forward(self, input):
        avg = input['timbre_avg']
        cov = input['timbre_cov']

        avg_coder = self.bn_avg(self.avg_encoder(avg) + avg)  # [B, 12]
        cov_coder = self.bn_cov(self.cov_encoder(cov) + cov)  # [B, 78]

        avg_coder = self.avg_encoder_layer(avg_coder)  # [B, 128]
        cov_coder = self.cov_encoder_layer(cov_coder)  # [B, 128]

        feature = torch.cat([avg_coder, cov_coder], dim=1)  # [B, 256]
        pred = self.classification_layer(feature)

        year = input['year'] - self.begin_year
        loss = self.compute_loss(pred, year)
        return pred, loss


class SY_Baseline(nn.Module):
    def __init__(self, num_years=90, begin_year=1922):
        super().__init__()

    def forward(self, input):
        pred = 0
        loss =0
        return pred, loss