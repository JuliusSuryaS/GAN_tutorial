import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class fcn8(nn.Module):
    def __init__(self, vgg, n_class):
        super(fcn8, self).__init__()
        self.conv1 = nn.Sequential(
            vgg.features[0],  # conv
            vgg.features[1],  # relu
            vgg.features[2],  # conv
            vgg.features[3],  # relu
            nn.MaxPool2d(2, 2, 0, 1, ceil_mode=True),
        )
        self.conv2 = nn.Sequential(
            vgg.features[5],  # conv
            vgg.features[6],  # relu
            vgg.features[7],  # conv
            vgg.features[8],  # relu
            nn.MaxPool2d(2, 2, 0, 1, ceil_mode=True),
        )
        self.conv3 = nn.Sequential(
            vgg.features[10],  # conv
            vgg.features[11],  # relu
            vgg.features[12],  # conv
            vgg.features[13],  # relu
            vgg.features[14],  # conv
            vgg.features[15],  # relu
            nn.MaxPool2d(2, 2, 0, 1, ceil_mode=True),
        )
        self.conv4 = nn.Sequential(
            vgg.features[17],  # conv
            vgg.features[18],  # relu
            vgg.features[19],  # conv
            vgg.features[20],  # relu
            vgg.features[21],  # conv
            vgg.features[22],  # relu
            nn.MaxPool2d(2, 2, 0, 1, ceil_mode=True),
        )
        self.conv5 = nn.Sequential(
            vgg.features[24],  # conv
            vgg.features[25],  # relu
            vgg.features[26],  # conv
            vgg.features[27],  # relu
            vgg.features[28],  # conv
            vgg.features[29],  # relu
            nn.MaxPool2d(2, 2, 0, 1, ceil_mode=True),
        )
        self.fc6 = nn.Sequential(
            nn.Conv2d(512, 4096, 1),
        )
        self.fc7 = nn.Sequential(
            nn.Conv2d(4096, n_class, 1),
        )
        self.fc7_16 = nn.Sequential(
            nn.Conv2d(512, n_class, 1),
        )
        self.fc7_8 = nn.Sequential(
            nn.Conv2d(256, n_class, 1),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out8 = self.conv3(out)
        out16 = self.conv4(out8)
        out32 = self.conv5(out16)
        out32 = self.fc6(out32)
        out32 = self.fc7(out32)
        out32_up = F.interpolate(out32, scale_factor=2, mode='bilinear')
        out16 = self.fc7_16(out16)
        out16 = out32_up + out16
        out16_up = F.interpolate(out16, scale_factor=2, mode='bilinear')
        out8 = self.fc7_8(out8)
        out = out8 + out16_up
        out = F.interpolate(out, scale_factor=8, mode='bilinear')
        return  out

class fcn16(nn.Module):
    def __init__(self, vgg, n_class):
        super(fcn16, self).__init__()
        self.conv1 = nn.Sequential(
            vgg.features[0],  # conv
            vgg.features[1],  # relu
            vgg.features[2],  # conv
            vgg.features[3],  # relu
            nn.MaxPool2d(2, 2, 0, 1, ceil_mode=True),
        )
        self.conv2 = nn.Sequential(
            vgg.features[5],  # conv
            vgg.features[6],  # relu
            vgg.features[7],  # conv
            vgg.features[8],  # relu
            nn.MaxPool2d(2, 2, 0, 1, ceil_mode=True),
        )
        self.conv3 = nn.Sequential(
            vgg.features[10],  # conv
            vgg.features[11],  # relu
            vgg.features[12],  # conv
            vgg.features[13],  # relu
            vgg.features[14],  # conv
            vgg.features[15],  # relu
            nn.MaxPool2d(2, 2, 0, 1, ceil_mode=True),
        )
        self.conv4 = nn.Sequential(
            vgg.features[17],  # conv
            vgg.features[18],  # relu
            vgg.features[19],  # conv
            vgg.features[20],  # relu
            vgg.features[21],  # conv
            vgg.features[22],  # relu
            nn.MaxPool2d(2, 2, 0, 1, ceil_mode=True),
        )
        self.conv5 = nn.Sequential(
            vgg.features[24],  # conv
            vgg.features[25],  # relu
            vgg.features[26],  # conv
            vgg.features[27],  # relu
            vgg.features[28],  # conv
            vgg.features[29],  # relu
            nn.MaxPool2d(2, 2, 0, 1, ceil_mode=True),
        )
        self.fc6 = nn.Sequential(
            nn.Conv2d(512, 4096, 1),
        )
        self.fc7 = nn.Sequential(
            nn.Conv2d(4096, n_class, 1),
        )
        self.fc7_16 = nn.Sequential(
            nn.Conv2d(512, n_class, 1),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out16 = self.conv4(out)
        out32 = self.conv5(out16)
        out32 = self.fc6(out32)
        out32 = self.fc7(out32)
        out32_up = F.interpolate(out32, scale_factor=2, mode='bilinear')
        out16 = self.fc7_16(out16)
        out = out32_up + out16
        out = F.interpolate(out, scale_factor=16, mode='bilinear')
        return out


class fcn32(nn.Module):
    def __init__(self, vgg, n_class):
        super(fcn32, self).__init__()
        self.conv1 = nn.Sequential(
            vgg.features[0], # conv
            vgg.features[1], # relu
            vgg.features[2], # conv
            vgg.features[3], # relu
            nn.MaxPool2d(2, 2, 0, 1, ceil_mode=True),
        )
        self.conv2 = nn.Sequential(
            vgg.features[5], # conv
            vgg.features[6], # relu
            vgg.features[7], # conv
            vgg.features[8], # relu
            nn.MaxPool2d(2, 2, 0, 1, ceil_mode=True),
        )
        self.conv3 = nn.Sequential(
            vgg.features[10], # conv
            vgg.features[11], # relu
            vgg.features[12], # conv
            vgg.features[13], # relu
            vgg.features[14], # conv
            vgg.features[15], # relu
            nn.MaxPool2d(2, 2, 0, 1, ceil_mode=True),
        )
        self.conv4 = nn.Sequential(
            vgg.features[17], # conv
            vgg.features[18], # relu
            vgg.features[19], # conv
            vgg.features[20], # relu
            vgg.features[21], # conv
            vgg.features[22], # relu
            nn.MaxPool2d(2, 2, 0, 1, ceil_mode=True),
        )
        self.conv5 = nn.Sequential(
            vgg.features[24], # conv
            vgg.features[25], # relu
            vgg.features[26], # conv
            vgg.features[27], # relu
            vgg.features[28], # conv
            vgg.features[29], # relu
            nn.MaxPool2d(2, 2, 0, 1, ceil_mode=True),
        )
        self. fc6 = nn.Sequential(
            nn.Conv2d(512, 4096, 1),
        )
        self.fc7 = nn.Sequential(
            nn.Conv2d(4096, n_class, 1),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.fc6(out)
        out = self.fc7(out)
        out = F.interpolate(out, scale_factor=32, mode='bilinear')
        return out

