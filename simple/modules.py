# ALL MODULES NEEDED IMPORTED HERE
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
