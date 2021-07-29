import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

class CNN(nn.Module):
    def __init__(self, in_channel = 3, n_actions=14):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size = 8, stride = 4)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        # self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)
        # self.bn3 = nn.BatchNorm2d(64)
        self.fc4 = nn.Linear(7*7*64, 512)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x.float()/255 # normalization
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)
