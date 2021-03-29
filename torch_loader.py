import os
import random
import torch
import numpy as np
from util import return_frames


class Dataset(torch.utils.data.Dataset):

    def __init__(self, opt, split_name):
        self.split_name = split_name
        self.k_shot = opt.k
        self.n_way = opt.n
        self.T = opt.T
        self.one_epoch_episode = opt.ep_num

    def random_sample_each_episode(self):
        dir = "dataset/" + self.split_name + "/"
        categories = os.listdir(dir)
        n_way_classes = random.sample(categories, self.n_way)

        query_x = torch.zeros((self.n_way, self.T, 3, 224, 224))
        query_y = torch.zeros((self.n_way, 1))

        support_x = torch.zeros((self.n_way * self.k_shot, self.T, 3, 224, 224,))
        support_y = torch.zeros((self.n_way * self.k_shot, 1))

        for n in range(self.n_way):
            index = random.randint(0, self.n_way - 1)
            class_name = n_way_classes[index]
            ex_dir = dir + class_name + "/"
            examples = os.listdir(ex_dir)
            query_x[n] = self._preprocess(ex_dir + random.sample(examples, 1)[0], self.split_name)
            query_y[n] = index
            for k in range(self.k_shot):
                class_name = n_way_classes[n]
                ex_dir = dir + class_name + "/"
                examples = os.listdir(ex_dir)
                support_x[n * self.k_shot + k] = self._preprocess(ex_dir + random.sample(examples, 1)[0],
                                                                  self.split_name)
                support_y[n * self.k_shot + k] = n
        return query_x, query_y, support_x, support_y

    def _preprocess(self, dir, split_type):

        frames = return_frames(dir, split_type)
        length = len(frames)
        sub_length = length // self.T
        l = np.array([random.randint(0, sub_length - 1) for i in range(self.T)])
        s = np.array([i * sub_length for i in range(self.T)])
        # print(l, s, self.T)
        indexes = l + s

        selected_frames = np.array(frames)[list(indexes)]
        return torch.from_numpy(selected_frames)

    def __len__(self):
        return self.one_epoch_episode

    def __getitem__(self, idx):

        query_x = torch.zeros((self.one_epoch_episode, self.n_way, self.T, 3, 224, 224))
        query_y = torch.zeros((self.one_epoch_episode, self.n_way, 1))

        support_x = torch.zeros((self.one_epoch_episode, self.n_way * self.k_shot, self.T, 3, 224, 224,))
        support_y = torch.zeros((self.one_epoch_episode, self.n_way * self.k_shot, 1))

        for i in range(self.one_epoch_episode):
            query_x[i], query_y[i], support_x[i], support_y[i] = self.random_sample_each_episode()

        return query_x, query_y, support_x, support_y

