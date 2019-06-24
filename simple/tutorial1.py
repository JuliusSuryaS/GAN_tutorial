import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def num_flat_features(self, x):
        size= x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2,2))

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



net = Net()
print(net)

params = list(net.parameters())
params_size = [p.size() for p in params]
# print(params_size)

inputs = torch.randn(1,1,32,32)
# out = net.forward(inputs)
out = net(inputs) # Network output
print(out)

# net.zero_grad()
# out.backward(torch.randn(1,10))

target = torch.randn(10)
target = target.view(1, -1) # netowrk fake ground truth

loss_criterion = nn.MSELoss()
loss = loss_criterion(out, target) # network loss

print(loss)

# Backprop visualization
# print(loss.grad_fn)
# print(loss.grad_fn.next_functions[0][0])
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

# BACKPROP
# Set optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

net.zero_grad() # clear gradient in buffer
loss.backward()
optimizer.step() # Update


