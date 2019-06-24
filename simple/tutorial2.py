import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using ', device)

# LOAD DATASET AND PREPROC
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


### Visualize data
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s'% classes[labels[i]] for i in range(4)))


# NETWORK
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5) # flatten
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def flatten(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        x = x.view(-1, num_features)
        return x

net = Net()
# IF STRICT IS SET TO FALSE, THEN IGNORE CURRENT THINGS NOT IN THE MODEL (load partial model)
net.load_state_dict(torch.load('./model/trained_model.pt'), strict=False) # LOAD SAVED MODEL
net.to(device) # assign to GPU
# If training
net.train()
# If evaluation
# net.eval() # handle things like batch norm and dropout
# print(net) # CPU
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001, momentum=0.9)

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs.to(device)) # assign to GPU
        # outputs = net(inputs) # CPU
        loss = criterion(outputs.to(device), labels.to(device)) # assign to GPU
        # loss = criterion(outputs, labels) # CPU
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 2000 == 1999:
            print('Step [%d/%d] loss : %.3f' % (epoch + 1, i + 1, running_loss/2000))
            running_loss = 0.0

    print('Training Completed')


# SAVE MODEL
torch.save(net.state_dict(),'./model/trained_model.pt') # SAVE MODEL



