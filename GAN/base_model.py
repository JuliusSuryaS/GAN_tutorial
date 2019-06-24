from base import *

class DCGAN_Generator(nn.Module):
    def __init__(self, nz=100, ngf=64):
        super(DCGAN_Generator, self).__init__()
        self.generator = nn.Sequential(
            # In 1x1x100
            nn.ConvTranspose2d(nz, 512, 4, 1, 0, bias=False), # out 4 x 4 x 512
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), # out 8 x 8x 256
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), # out 16 x 16 x 128
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), # out 32 x 32 x 64
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False), # out 64 x 64 x 3
            nn.Tanh()
        )

    def forward(self, x):
        out = self.generator(x)
        return out

class DCGAN_Discriminator(nn.Module):
    def __init__(self, nz=3, ndf=64):
        super(DCGAN_Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(nz, 64, 4, 2, 1, bias=False), # out 32 x 32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), # out 16 x 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), # out 8 x 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), # out 4 x 4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False), # out 1 x 1
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.discriminator(x)
        return out

class CGAN1(nn.Module):
    def __init__(self, nz=3):
        super(CGAN1, self).__init__()
        # self.conv1 = nn.ConvTranspose2d(nz, 512, 3, 1, 1, bias=False) # out 4 x 4 x 512
        # self.conv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False) # out 8 x 8x 256
        # self.bn2 = nn.BatchNorm2d(256)
        # self.conv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False) # out 16 x 16 x 128
        # self.bn3 = nn.BatchNorm2d(128)
        # self.conv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False) # out 32 x 32 x 64
        # self.bn4 = nn.BatchNorm2d(64)
        # self.conv5 = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False) # out 64 x 64 x 3

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(3, 512, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        ) # 8 x 8
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(512 + 3, 256, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        ) # 16 x 16
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(256 + 3,128, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ) # 32 x 32
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(128 + 3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ) # 64 x 64
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(64 + 3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, 1, 1),
            nn.Tanh(),
        ) # 64 x 64

    def forward(self, x):
        x4 = F.interpolate(x, (4,4), mode='bilinear')
        x3 = F.interpolate(x, (8,8), mode='bilinear')
        x2 = F.interpolate(x, (16,16), mode='bilinear')
        x1 = F.interpolate(x, (32,32), mode='bilinear')
        x0 = x

        out1 = self.conv1(x4) # 8 x 8
        out2 = self.conv2(torch.cat((out1,x3), 1))
        out3 = self.conv3(torch.cat((out2,x2), 1))
        out4 = self.conv4(torch.cat((out3,x1), 1))
        out5 = self.conv5(torch.cat((out4,x0), 1))
        return out5

class MGenerator1(nn.Module):
    def __init__(self):
        super(MGenerator1, self).__init__()
        self.block1 = conv_sn_relu(3, 64, 3, 1, 1) # 128 x 256
        self.block2 = conv_sn_relu(64, 128, 4, 2, 1) # 64  x 128
        self.block3 = conv_sn_relu(128, 256, 4, 2, 1) # 32  x 64
        self.block4 = conv_sn_relu(256, 512, 4, 2, 1) # 16  x 32
        self.block5 = conv_sn_relu(512, 512, 4, 2, 1) # 8  x 16

        self.dblock0 = convT_sn_leak(512, 512, 3, 1, 1) # 8 x 16

        self.dblock1 = convT_sn_leak(512 +  512, 512, 4, 2, 1) # 16 x 32
        self.dblock2 = convT_sn_leak(512 + 512, 512, 4, 2, 1) # 32 x 64
        self.dblock3 = convT_sn_leak(512 + 256, 256, 4, 2, 1) # 64 x 128
        self.dblock4 = convT_sn_leak(256 + 256, 128, 4, 2, 1) # 128 x 256
        self.dblock5 = nn.ConvTranspose2d(128 + 64, 3, 3, 1, 1) # 128 x 256

    def forward(self, x):
        enc1 = self.block1(x)
        enc2 = self.block2(enc1)
        enc3 = self.block3(enc2)
        enc4 = self.block4(enc3)
        enc5 = self.block5(enc4)
        dec0 = self.dblock0(enc5)
        dec1 = self.dblock1(dstack(dec0,enc5))
        dec2 = self.dblock2(dstack(dec1,enc4))
        dec3 = self.dblock3(dstack(dec2,enc3))
        dec4 = self.dblock4(dstack(dec3,enc2))
        dec5 = F.tanh(self.dblock5(dstack(dec4,enc1)))
        return dec5

class MDiscriminator1(nn.Module):
    def __init__(self, use_sigmoid=True):
        super(MDiscriminator1, self).__init__()
        self.block1 = conv_sn_leak(3, 128, 3, 1, 1) # 128 x 256
        self.block2 = conv_sn_leak(128, 256, 4, 2, 1) # 64 x 128
        self.block3 = conv_sn_leak(256, 512, 4, 2, 1) # 32 x 64
        self.block4 = conv_sn_leak(512, 512, 4, 2, 1) # 16 x 32
        self.block5 = conv_sn_leak(512, 512, 4, 2, 1) # 8 x 16
        self.block6 = conv_sn_leak(512, 1024, 4, 2, 1) # 4 x 8
        self.block7 = conv_sn_leak(1024, 1024, 4, 2, 1) # 2 x 4

        if use_sigmoid:
            self.block8 = nn.Sequential(
                nn.Conv2d(1024, 1, (2,4), 1, 0), # 1 x 1
                nn.Sigmoid(),
            )
        else:
            self.block8 = nn.Conv2d(1024, 1, (2,4),1, 0) # 1 x 1

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block8(out)
        return out


class conv_sn_relu(nn.Module):
    def __init__(self, dim_in, dim_out, ksz=3, s=1, pad=1):
        super(conv_sn_relu, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_out, ksz, s, pad)

    def forward(self, x):
        out = F.relu((self.conv1(x)))
        return out

class conv_sn_leak(nn.Module):
    def __init__(self, dim_in, dim_out, ksz=3, s=1, pad=1):
        super(conv_sn_leak, self).__init__()
        self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_out, ksz, s, pad))

    def forward(self, x):
        out = F.leaky_relu((self.conv1(x)))
        return out

class convT_sn_leak(nn.Module):
    def __init__(self, dim_in, dim_out, ksz=3, s=1, pad=1):
        super(convT_sn_leak, self).__init__()
        self.conv1 = spectral_norm(nn.ConvTranspose2d(dim_in, dim_out, ksz, s, pad))

    def forward(self, x):
        out = F.leaky_relu((self.conv1(x)))
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def dstack(x,y, dim=1):
    return torch.cat((x,y), dim=1)