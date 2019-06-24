import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.txt_path = 'dataset/train.txt'
        self.imgs_path = 'dataset/img'
        self.segs_path = 'dataset/cls'
        self.data_list = self.read_data_list()
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_idx = self.data_list[idx]
        img_file =  os.path.join(self.root_dir, self.imgs_path, data_idx + '.jpg')
        seg_file =  os.path.join(self.root_dir, self.segs_path, data_idx + '.mat')

        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        seg = sio.loadmat(seg_file)['GTcls']['Segmentation'][0][0]

        sample = {'img': img, 'seg': seg}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def read_data_list(self):
        txt_file = os.path.join(self.root_dir, self.txt_path)
        im_list = []
        with open(txt_file) as f:
           line = f.readlines()
        for l in line:
            im_list.append(l.strip())
        return im_list


class ProcData():
    def __call__(self, sample):
        img, seg = sample['img'], sample['seg']

        img = cv2.resize(img, (224, 224))
        seg = cv2.resize(seg, (224, 224))

        img = torch.from_numpy(img.transpose(2,0,1))
        # seg = torch.from_numpy(seg.transpose(2,0,1))
        seg = torch.from_numpy(seg).long()

        img = img.type(torch.FloatTensor) / 255.
        img[0,:,:].add_(0.485).div_(0.229)
        img[1,:,:].add_(0.456).div_(0.224)
        img[2,:,:].add_(0.406).div_(0.225)


        return {'img': img, 'seg': seg}