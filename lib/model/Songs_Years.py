import math
import torch
import torch.nn as nn
from lib.utils.logger import logger
from lib.utils.misc import param_size


def emb1d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature**(2 * (torch.div(dim_t, 2, rounding_mode='floor')) / num_pos_feats)
    posemb = pos/ dim_t
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
        self.avg_encoder = nn.Sequential(nn.Linear(12, 128), nn.ReLU(), nn.Linear(128, 12))
        self.cov_encoder = nn.Sequential(nn.Linear(78, 128), nn.ReLU(), nn.Linear(128, 78))
        self.drop_encoder_avg = nn.Dropout(0.1)
        self.drop_encoder_cov = nn.Dropout(0.1)
        self.drop_cov = nn.Dropout(0.1)
        self.avg_encoder_layer = nn.Sequential(nn.Linear(12, 128), nn.ReLU(), nn.Linear(128, 128))
        self.cov_encoder_layer = nn.Sequential(nn.Linear(78, 128), nn.ReLU(), nn.Linear(128, 128))
        self.drop = nn.Dropout(0.1)
        self.classification_layer = nn.Sequential(nn.Linear(256, 128), nn.ReLU(),
                                                  nn.Linear(128, 128), nn.ReLU(),
                                                  nn.Linear(128, num_years))
        self.compute_loss = nn.CrossEntropyLoss()
        logger.info(f"{self.name} has {param_size(self)}M parameters")

    def forward(self, input):
        avg = input['timbre_avg']
        cov = input['timbre_cov']

        avg_coder = self.bn_avg(self.drop_avg(self.avg_encoder(avg)) + avg)  # [B, 12]
        cov_coder = self.bn_cov(self.drop_cov(self.cov_encoder(cov)) + cov)  # [B, 78]

        avg_coder = self.drop_encoder_avg(self.avg_encoder_layer(avg_coder))  # [B, 128]
        cov_coder = self.drop_encoder_cov(self.cov_encoder_layer(cov_coder))  # [B, 128]

        feature = torch.cat([avg_coder, cov_coder], dim=1)  # [B, 256]
        pred = self.drop(self.classification_layer(feature))

        year = input['year'] - self.begin_year
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
        self.embed = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 128))

        self.attn1 = nn.MultiheadAttention(128, 8, dropout=0.1)
        self.drop1 = nn.Dropout(0.1)
        self.norm1 = nn.BatchNorm1d(128)
        self.attn2 = nn.MultiheadAttention(128, 8, dropout=0.1)
        self.drop2 = nn.Dropout(0.1)
        self.norm2 = nn.BatchNorm1d(128)

        self.ffn = nn.Sequential(nn.Linear(128, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(),
                                 nn.Linear(64, num_years))
        self.drop3 = nn.Dropout(0.1)
        self.norm3 = nn.BatchNorm1d(num_years)

        self.decoder = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, num_years))
        self.compute_loss = nn.CrossEntropyLoss()
        logger.info(f"{self.name} has {param_size(self)}M parameters")

    def encoder_layer(self, x):
        x = emb1d(self.embed(x))
        x = x + self.drop1(self.attn1(x, x, x)[0])
        x = self.norm1(x)
        x = x + self.drop2(self.attn2(x, x, x)[0])
        x = self.norm2(x)
        x = self.drop3(self.ffn(x))
        x = self.norm3(x)
        return x

    def forward(self, input):
        if self.mode == 'avg':
            x = input['timbre_avg']
        elif self.mode == 'cov':
            x = input['timbre_cov']
        else:
            x = torch.cat([input['timbre_avg'], input['timbre_cov']], dim=1)

        x = self.encoder_layer(x)
        # pred = self.decoder(x)

        year = input['year'] - self.begin_year
        loss = self.compute_loss(x, year)
        return x, loss, x