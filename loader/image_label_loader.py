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

class imageLabelLoader(data.Dataset):
    def __init__(self, root, dataName, phase='train', img_size=(241,121)):
        self.root = root
        self.dataName = dataName
        self.phase = phase
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([128, 128, 128])
        self.files = collections.defaultdict(list)
        """
        for phase in ['train', 'val', 'train+5light']:
            file_list = tuple(open(root +'/' + dataName +'/'+ phase + '.txt', 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[phase] = file_list
        """
        file_list = tuple(open(root + '/' + dataName + '/' + self.phase + '.txt', 'r'))
        file_list = [id_.rstrip() for id_ in file_list]
        self.files[self.phase] = file_list
        # print self.files['train']
    def __len__(self):
        return len(self.files[self.phase])

    def __getitem__(self, index):
        img_name = self.files[self.phase][index]
        img_path = self.root + '/' + self.dataName + '/' + 'image/' + self.phase+'/'+img_name + '.jpg'
        lbl_path = self.root+ '/' + self.dataName + '/' + 'label/' + self.phase+'/'+img_name + '.png'

        img = Image.open(img_path)
        #if img.shape
        img_size = img.size
        if self.img_size[1] != img_size[0] or self.img_size[0] != img_size[1]:
            img = img.resize((self.img_size[1], self.img_size[0]), resample=Image.BILINEAR)
        img = np.array(img, dtype=np.float32)
        if not (len(img.shape) == 3 and img.shape[2] == 3):
            img = img.reshape(img.shape[0], img.shape[1], 1)
            img = img.repeat(3, 2)
        img -= self.mean
        img = img[:, :, ::-1]# RGB -> BGR
        img = img.transpose(2, 0, 1)
        img = img.copy()

        lbl = Image.open(lbl_path)
        lbl_size = lbl.size

        if self.img_size[1] != lbl_size[0] or self.img_size[0] != lbl_size[1]:
            lbl = lbl.resize((self.img_size[1], self.img_size[0]), resample=Image.BILINEAR)

        lbl = np.array(lbl)
        lbl = lbl.copy()

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl