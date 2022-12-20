import torch
import argparse
from lib.utils.etqdm import etqdm
from lib.dataset.Songs import Songs
from lib.utils.misc import bar_perfixes
from torch.utils.tensorboard import SummaryWriter
from lib.utils.logger import logger
from lib.model.Songs_Years import Songs_Years


def ANN_worker():

    model = Songs_Years()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=100)

    arg = parser.parse_args()

    logger.set_log_file('exp/ANN/dev', 'dev')
    Songs()
    ANN_worker()
