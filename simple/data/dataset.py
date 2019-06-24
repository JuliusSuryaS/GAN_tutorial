import os
import glob
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2

# Dataset maker class
class PanoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return 5

    def _imRead(self, x):
        image = cv2.imread(x)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, idx):
        # can read image in any way <ONLY NEED FOLDER STRUCTURED WITH NUMBER>
        img_name_gt = glob.glob(self.root_dir + '*_' + str(idx + 1) + '/gt_1.jpg')
        img_name = glob.glob(self.root_dir + '*_' + str(idx + 1) + '/im_1.jpg')

        label_img = self._imRead(img_name_gt[0])
        # img_name = os.path.join(self.root_dir, '*_' + str(idx+1), 'im_1.jpg')
        input_img = self._imRead(img_name[0])

        # can be dict or list or tuple or anything
        sample = {'input': input_img, 'label': label_img}

        if self.transform:
            sample = self.transform(sample)
        return sample

# Dataset transform fn class
class ToTensor(object):
    def __call__(self, sample):
        input_img, label_img = sample['input'], sample['label']
        input_img = input_img.transpose((2,0,1))
        label_img = label_img.transpose((2,0,1))
        # return must follow Dataset Class output
        return {'input': torch.from_numpy(input_img), 'label': torch.from_numpy(label_img)}

# Dataset transform fn class
class ResizeImg(object):
    def __call__(self, sample):
        input_img, label_img = sample['input'], sample['label']
        input_img = cv2.resize(input_img, (128, 128))
        label_img = cv2.resize(label_img, (128, 128))
        # return must follow Dataset Class output
        return {'input': input_img, 'label': label_img}


# Basic DATA Definition
data_path = '/home/juliussurya/work/360dataset/pano_cropped_outdoor_rand/'
# pano_dataset = PanoDataset(root_dir=data_path, transform=False)

# for i in range(len(pano_dataset)):
#     sample = pano_dataset[i]
#     image = sample['input']
#     label = sample['label']
#
#     plt.imshow(image)
#     plt.show()
#     plt.imshow(label)
#     plt.show()

transform_fn = transforms.Compose([ResizeImg(), ToTensor()]) # define data transform function
proc_dataset = PanoDataset(root_dir=data_path, transform=transform_fn) # make dataset

# BASELINE METHOD
# for i in range(len(proc_dataset)):
#    sample = proc_dataset[i]
#    print(sample['input'].size())

# BETTER METHOD
# Pytorch data laoder
dataloader = DataLoader(proc_dataset, batch_size=2, shuffle=True, num_workers=2)

for i, data_next in enumerate(dataloader):
    print(i, data_next['input'].size(), data_next['label'].size())