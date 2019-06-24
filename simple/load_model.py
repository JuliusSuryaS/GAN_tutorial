import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import Networks as Networks

model = Networks.Net()
print(model) # Print network summary

# 1. Saving model after training
model.train() # set model as training model (batchnorm, dropout, etc)
# Training ...

# SAVE CHECKPOINT
optimizer = optim.SGD(model.parameters(), lr=0.04)
# on running epoch
epoch = 10
loss = 0

# TRAINING LOOP ETC
# SAVE MANY THINGS and others
torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, './model/ckpt.pt')

# LOAD MANY THINGS DURING CHECKPOINT
checkpoint = torch.load('./model/ckpt.pt')
model.load_state_dict(checkpoint['model_state_dict']) # load desired stuff one by on
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']



# Save Model
# torch.save(model.state_dict(), './model/trained_model.pt')

# 2. Load model
pretrained_model = torch.load('./model/trained_model.pt')
model.eval() # set model to inference mode (disable batchnorm, dropout, etc)

# 2.1. Check what's in pretrained_model
for key, val in pretrained_model.items():
    print(key, val.size())
    # Do some operations to remove unwanted weights ...

model.load_state_dict(pretrained_model) # load the weights to model


# ============================================================
#                      Different Model
# ============================================================
model = Networks.NetModified() # load different model
# model.load_state_dict(pretrained_model) # load --> will be error
model.load_state_dict(pretrained_model, strict=False)
# or
# Modified the dictiionary to match current model



