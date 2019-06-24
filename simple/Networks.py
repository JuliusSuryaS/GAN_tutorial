import torch.nn as nn
import torch.nn.functional as F

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

class NetModified(nn.Module):
    def __init__(self):
        super(NetModified, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2Mod = nn.Conv2d(6, 16, 5) # change the name so weights is different
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
