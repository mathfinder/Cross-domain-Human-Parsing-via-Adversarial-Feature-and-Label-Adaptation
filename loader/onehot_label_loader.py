import os
import collections
import json
import torch
import torchvision
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import Image
from tqdm import tqdm
from torch.utils import data

class onehotLabelLoader(data.Dataset):
    def __init__(self, root, dataName, phase='train', label_nums=12, lbl_size=(241,121)):
        self.root = root
        self.dataName = dataName
        self.phase = phase
        self.label_nums = label_nums
        self.lbl_size = lbl_size if isinstance(lbl_size, tuple) else (lbl_size, lbl_size)
        self.files = collections.defaultdict(list)
        self.now_idx = 0
        file_list = tuple(open(root + '/' + dataName + '/' + self.phase + '.txt', 'r'))
        file_list = [id_.rstrip() for id_ in file_list]
        self.files[self.phase] = file_list
        # print self.files['train']
    def __len__(self):
        return len(self.files[self.phase])

    def __getitem__(self, index):
        lbl_name = self.files[self.phase][index]
        lbl_path = self.root + '/' + self.dataName + '/' + 'label/' + self.phase + '/' + lbl_name + '.png'

        lbl = Image.open(lbl_path)
        lbl_size = lbl.size
        if self.lbl_size[1] != lbl_size[0] or self.lbl_size[0] != lbl_size[1]:
            lbl = lbl.resize((self.lbl_size[1], self.lbl_size[0]), resample=Image.BILINEAR)

        lbl = np.array(lbl, dtype=np.float32)
        lbl_onehot = np.zeros((self.label_nums, self.lbl_size[0], self.lbl_size[1]))
        for i in range(self.label_nums):
            lbl_onehot[i][lbl==i] = 1

        lbl = lbl.copy()

        lbl = torch.from_numpy(lbl).float()
        lbl_onehot = torch.from_numpy(lbl_onehot).float()

        return lbl, lbl_onehot

    def getBatch(self, index):
        self.now_idx = (self.now_idx + 1)%len()
        lbl_name = self.files[self.phase][index]
        lbl_path = self.root + '/' + self.dataName + '/' + 'label/' + self.phase + '/' + lbl_name + '.png'

        lbl = Image.open(lbl_path)
        lbl_size = lbl.size
        if self.lbl_size[1] != lbl_size[0] or self.lbl_size[0] != lbl_size[1]:
            lbl = lbl.resize((self.lbl_size[1], self.lbl_size[0]), resample=Image.BILINEAR)

        lbl = np.array(lbl, dtype=np.float32)
        lbl_onehot = np.zeros((self.label_nums, self.lbl_size[0], self.lbl_size[1]))
        for i in range(self.label_nums):
            lbl_onehot[i][lbl == i] = 1

        lbl = lbl.copy()

        lbl = torch.from_numpy(lbl).float()
        lbl_onehot = torch.from_numpy(lbl_onehot).float()

        return lbl, lbl_onehot
