import torch
import torch.nn as nn
import torchvision.models as models

vgg = models.vgg16(pretrained=True)
model = vgg.features

print(model.children)

# model_mine = nn.Sequential(*list(model.children())[0:4])
# print(model_mine)


# SLICE VGG (same with pix2pixhd)
slice1 = torch.nn.Sequential()
slice2 = torch.nn.Sequential()
for i in range(2):
    slice1.add_module(str(i), model[i])
for i in range(2,7):
    slice2.add_module(str(i), model[i])


print(slice1)
print(slice2)



class CustModel(nn.Module):
    def __init__(self):
        super(CustModel, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.conv1 = vgg[0]
        self.conv2 = vgg[2]

    def forward(self, x):
        x = self.conv1(x)
        x  = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.tanh(x)
        return x


custModel = CustModel()
print(custModel)

