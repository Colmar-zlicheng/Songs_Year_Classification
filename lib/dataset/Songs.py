import os
import pickle
import torch
import torch.nn as nn
import pandas as pd
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from termcolor import colored


class Songs(torch.utils.data.Dataset):

    def __init__(self, data_split='train', train_size=2000, test_size=200, seed=0):
        super().__init__()
        self.root = './data'
        self.path = './data/songs.csv'
        self.data_split = data_split
        self.data = self.Songs_whole(train_size=train_size, test_size=test_size, seed=seed)

    def Songs_whole(self, train_size=2000, test_size=200, seed=0):
        # 1969-2010: train_size + test_size for each year
        # train_size + test_size should less than 2200
        cache_name = f"data_dict_{train_size}+{test_size}.pkl"
        cache_path = os.path.join(self.root, cache_name)
        torch.manual_seed(seed)
        if not os.path.exists(cache_path):
            logger.info("Cache not exist, beginning generate and save data into {}".format(cache_path))
            data_raw = pd.read_csv(self.path)
            year = data_raw['year']
            year = torch.tensor(year)
            _, count = torch.unique(year, return_counts=True)
            # years, index, count = torch.unique(year, return_inverse=True, return_counts=True)

            year_train = []
            year_test = []
            timbre_avg_train = []
            timbre_avg_test = []
            timbre_cov_train = []
            timbre_cov_test = []
            year_bar = etqdm(range(1969, 2010 + 1))
            for y in year_bar:
                index_y = torch.nonzero(year == y)
                # -1923 is for correct index
                index_y = index_y[torch.randint(0, count[y-1923], (train_size+test_size,))]

                year_y = year[index_y]
                year_train.append(year_y[:train_size].reshape(train_size))  # append [2000]
                year_test.append(year_y[train_size:].reshape(test_size))  # append [200]

                timbre_avg_train_y = []
                timbre_avg_test_y = []
                for i_avg in range(12):
                    year_bar.set_description(f"year:{colored(f'{y}', 'white', attrs=['bold'])}, avg:{i_avg}")
                    timbre_avg_i = f"timbre_avg_{i_avg}"
                    timbre_avg_y = torch.tensor(data_raw[timbre_avg_i])[index_y]  # [train_size+test_size, 1]
                    timbre_avg_train_y.append(timbre_avg_y[:train_size])
                    timbre_avg_test_y.append(timbre_avg_y[train_size:])
                # timbre_avg_train_y: [train_size, 12], timbre_avg_test_y: [test_size, 12]
                timbre_avg_train_y = torch.stack(timbre_avg_train_y, dim=0).reshape(12, train_size).transpose(0, 1)
                timbre_avg_test_y = torch.stack(timbre_avg_test_y, dim=0).reshape(12, test_size).transpose(0, 1)
                timbre_avg_train.append(timbre_avg_train_y)
                timbre_avg_test.append(timbre_avg_test_y)

                timbre_cov_train_y = []
                timbre_cov_test_y = []
                for i_cov in range(78):
                    year_bar.set_description(f"year:{colored(f'{y}', 'white', attrs=['bold'])}, cov:{i_cov}")
                    timbre_cov_i = f"timbre_cov_{i_cov}"
                    timbre_cov_y = torch.tensor(data_raw[timbre_cov_i])[index_y]
                    timbre_cov_train_y.append(timbre_cov_y[:train_size])
                    timbre_cov_test_y.append(timbre_cov_y[train_size:])
                # timbre_cov_train_y: [train_size, 78], timbre_cov_test_y: [test_size, 78]
                timbre_cov_train_y = torch.stack(timbre_cov_train_y, dim=0).reshape(78, train_size).transpose(0, 1)
                timbre_cov_test_y = torch.stack(timbre_cov_test_y, dim=0).reshape(78, test_size).transpose(0, 1)
                timbre_cov_train.append(timbre_cov_train_y)
                timbre_cov_test.append(timbre_cov_test_y)

            year_train = torch.stack(year_train, dim=0).reshape(42*train_size)
            year_test = torch.stack(year_test, dim=0).reshape(42*test_size)
            timbre_avg_train = torch.stack(timbre_avg_train, dim=0).reshape(42*train_size, 12)
            timbre_avg_test = torch.stack(timbre_avg_test, dim=0).reshape(42*test_size, 12)
            timbre_cov_train = torch.stack(timbre_cov_train, dim=0).reshape(42*train_size, 78)
            timbre_cov_test = torch.stack(timbre_cov_test, dim=0).reshape(42*test_size, 78)

            data_whole = dict()
            data_whole['timbre_avg_train'] = timbre_avg_train  # [42*train_size, 12]
            data_whole['timbre_avg_test'] = timbre_avg_test  # [42*test_size, 12]
            data_whole['timbre_cov_train'] = timbre_cov_train  # [42*train_size, 78]
            data_whole['timbre_cov_test'] = timbre_cov_test  # [42*test_size, 78]
            data_whole['year_train'] = year_train  # [42*train_size]
            data_whole['year_test'] = year_test  # [42*test_size]
            with open(cache_path, 'wb') as f:
                pickle.dump(data_whole, f)
        else:
            logger.info("Load Data from cache:{} with train_size:{} and test_size:{}"
                        .format(cache_path, train_size, test_size))
            with open(cache_path, 'rb') as ff:
                data_whole = pickle.load(ff)
        return data_whole

    def __len__(self):
        if self.data_split == 'train':
            self.len = len(self.data['year_train'])
        else:
            self.len = len(self.data['year_test'])
        return self.len

    def __getitem__(self, item):
        sample = dict()
        if self.data_split == 'train':
            sample['timbre_avg'] = self.data['timbre_avg_train'][item]
            sample['timbre_cov'] = self.data['timbre_cov_train'][item]
            sample['year'] = self.data['year_train'][item]
        else:
            sample['timbre_avg'] = self.data['timbre_avg_test'][item]
            sample['timbre_cov'] = self.data['timbre_cov_test'][item]
            sample['year'] = self.data['year_test'][item]
        return sample
