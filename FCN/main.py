import os
import sys
import time
import random
import matplotlib.pyplot as plt
from customdataset import CustomDataset, ProcData
from model import fcn8, fcn16, fcn32

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# =======================================
# Parameters
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_root = '/home/juliussurya/work/VOC/benchmark_RELEASE'
batch_sz = 16
shuffle = True
workers = 1
learn_rate = 0.0002
total_epochs = 10

# Init dataset
tsfm_fn = transforms.Compose([ProcData()])
VOCdata = CustomDataset(data_root, transform=tsfm_fn)
data_loader = DataLoader(VOCdata, batch_size=batch_sz, shuffle=shuffle, num_workers=workers)

# Network settings
# Load vgg network
vgg16 = torchvision.models.vgg16(pretrained=False)
weights = torch.load('./weights/vgg16-397923af.pth')
vgg16.load_state_dict(weights)

# Init FCN
net = fcn8(vgg16, 21)
net.to(device)
net.train()
print(net)
input('....')

# Define network loss
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learn_rate, betas=(0.5,0.99))

# Summary
writer = SummaryWriter('./log')

# Training Loop
for epoch in range(total_epochs):
    for itr, batch in enumerate(data_loader):
        # Clear gradient
        net.zero_grad()

        # load data
        img, seg = batch['img'].to(device), batch['seg'].to(device)

        # forward
        out = net(img)

        # compute loss
        loss = loss_fn(out, seg)

        # optimize
        loss.backward()
        optimizer.step()

        if itr % 100 == 0:
            print('%d | Iter %d : Loss %.4f' %(epoch, itr, loss.item()))
            writer.add_scalar('Loss', loss, itr)
            out_im = torch.argmax(out, 1, keepdim=True)
            gt = seg.unsqueeze(1)
            out_file = './out/' + 'epoch_' + str(epoch) + '/out_' + str(itr) + '.jpg'
            im_file = './out/' + 'epoch_' + str(epoch) + '/img_' + str(itr) + '.jpg'
            gt_file = './out/' + 'epoch_' + str(epoch) + '/gt_' + str(itr) + '.jpg'
            torchvision.utils.save_image(out_im, out_file, normalize=False)
            torchvision.utils.save_image(img, im_file, normalize=False)
            torchvision.utils.save_image(gt, gt_file, normalize=False)

    # Save weights every epoch
    model_path = './weights/latest_trained.pth'
    torch.save(net.state_dict(), model_path)
