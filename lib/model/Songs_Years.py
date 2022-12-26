import math
import torch
import torch.nn as nn
from lib.utils.logger import logger
from lib.utils.misc import param_size


def emb1d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (torch.div(dim_t, 2, rounding_mode='floor')) / num_pos_feats)
    posemb = pos / dim_t
    posemb = torch.stack((posemb[..., 0::2].sin(), posemb[..., 1::2].cos()), dim=-1).flatten(-2)
    return posemb


class Songs_Years(nn.Module):

    def __init__(self, num_years=42, begin_year=1969):
        super().__init__()
        self.name = type(self).__name__
        self.begin_year = begin_year
        self.bn_avg = nn.BatchNorm1d(12)
        self.bn_cov = nn.BatchNorm1d(78)
        self.drop_avg = nn.Dropout(0.1)
        self.drop_cov = nn.Dropout(0.1)
        self.avg_encoder = nn.Sequential(nn.Linear(12, 128), nn.ReLU(),
                                         nn.Linear(128, 128), nn.ReLU(),
                                         nn.Linear(128, 12))
        self.cov_encoder = nn.Sequential(nn.Linear(78, 128), nn.ReLU(),
                                         nn.Linear(128, 128), nn.ReLU(),
                                         nn.Linear(128, 78))
        self.drop_encoder_avg = nn.Dropout(0.1)
        self.drop_encoder_cov = nn.Dropout(0.1)
        self.avg_encoder_layer = nn.Sequential(nn.Linear(12, 128), nn.ReLU(),
                                               nn.Linear(128, 128), nn.ReLU(),
                                               nn.Linear(128, 256))
        self.cov_encoder_layer = nn.Sequential(nn.Linear(78, 128), nn.ReLU(),
                                               nn.Linear(128, 128), nn.ReLU(),
                                               nn.Linear(128, 256))
        self.drop = nn.Dropout(0.1)
        self.classification_layer = nn.Sequential(nn.Linear(512, 512), nn.ReLU(),
                                                  nn.Linear(512, 256), nn.ReLU(),
                                                  nn.Linear(256, 128), nn.ReLU(),
                                                  nn.Linear(128, num_years))
        self.compute_loss = nn.CrossEntropyLoss()
        logger.info(f"{self.name} has {param_size(self)}M parameters")

    def forward(self, inputs):
        avg = inputs['timbre_avg']
        cov = inputs['timbre_cov']

        avg_coder = self.bn_avg(self.drop_avg(self.avg_encoder(avg)) + avg)  # [B, 12]
        cov_coder = self.bn_cov(self.drop_cov(self.cov_encoder(cov)) + cov)  # [B, 78]

        avg_coder = self.drop_encoder_avg(self.avg_encoder_layer(avg_coder))  # [B, 256]
        cov_coder = self.drop_encoder_cov(self.cov_encoder_layer(cov_coder))  # [B, 256]

        feature = torch.cat([avg_coder, cov_coder], dim=1)  # [B, 512]
        pred = self.drop(self.classification_layer(feature))

        year = inputs['year'] - self.begin_year
        loss = self.compute_loss(pred, year)
        return pred, loss


class SY_Baseline(nn.Module):

    def __init__(self, num_years=90, begin_year=1922, mode='avg'):
        super().__init__()
        self.name = type(self).__name__
        self.mode = mode
        self.begin_year = begin_year
        if self.mode == 'avg':
            input_dim = 12
        elif self.mode == 'cov':
            input_dim = 78
        else:
            input_dim = 90
        self.begin_year = begin_year
        self.bn_x = nn.BatchNorm1d(input_dim)
        self.drop_x = nn.Dropout(0.1)
        self.x_encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(),
                                         nn.Linear(128, 128), nn.ReLU(),
                                         nn.Linear(128, input_dim))

        self.drop_encoder_x = nn.Dropout(0.1)
        self.x_encoder_layer = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(),
                                               nn.Linear(128, 128), nn.ReLU(),
                                               nn.Linear(128, 256))
        self.drop = nn.Dropout(0.1)
        self.classification_layer = nn.Sequential(nn.Linear(256, 256), nn.ReLU(),
                                                  nn.Linear(256, 128), nn.ReLU(),
                                                  nn.Linear(128, num_years))
        self.compute_loss = nn.CrossEntropyLoss()
        logger.info(f"{self.name} has {param_size(self)}M parameters")

    def forward(self, inputs):
        if self.mode == 'avg':
            x = inputs['timbre_avg']
        elif self.mode == 'cov':
            x = inputs['timbre_cov']
        else:
            x = torch.cat([inputs['timbre_avg'], inputs['timbre_cov']], dim=1)

        x_coder = self.bn_x(self.drop_x(self.x_encoder(x)) + x)  # [B, 12]

        x_coder = self.drop_encoder_x(self.x_encoder_layer(x_coder))  # [B, 256]

        pred = self.drop(self.classification_layer(x_coder))

        year = inputs['year'] - self.begin_year
        loss = self.compute_loss(pred, year)
        return pred, loss
