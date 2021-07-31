import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

import numpy as np

class CNN(nn.Module):
    def __init__(self, in_channel, n_actions):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size = 8, stride = 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc4 = nn.Linear(7*4*64, 512)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # print(x.shape)
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)

    # Automatically find the feature shape at the end of convolution layer.
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)