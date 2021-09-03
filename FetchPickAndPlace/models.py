import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# 보통 Q-function을 parameterize할 때 observation을 input으로 받고 output을 action space dimension에 맞게 한다.
# 하지만 continuous space의 deterministic action은 그게 불가능하다.

# Batch normalization 효과는 생각보다 큰 것 같다.

class Qnet(nn.Module):
    def __init__(self, observation_dim, action_space, l1_channel = 256, l2_channel = 128):
        super(Qnet, self).__init__()
        self.fc_s = nn.Linear(observation_dim, l1_channel)
        self.fc_a = nn.Linear(l1_channel + action_space.shape[0], l2_channel)
        self.fc_out = nn.Linear(l2_channel, 1)

        # f1 = 1./np.sqrt(self.fc_s.weight.data.size()[0])
        # nn.init.uniform_(self.fc_s.weight.data, -f1, f1)
        # nn.init.uniform_(self.fc_s.bias.data, -f1, f1)
        #
        # f2 = 1./np.sqrt(self.fc_a.weight.data.size()[0])
        # nn.init.uniform_(self.fc_a.weight.data, -f2, f2)
        # nn.init.uniform_(self.fc_a.bias.data, -f2, f2)
        #
        # f3 = 0.003 # specified in the paper
        # nn.init.uniform_(self.fc_out.weight.data, -f3, f3)
        # nn.init.uniform_(self.fc_out.bias.data, -f3, f3)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(torch.cat([h1,a],dim=1)))
        q = F.relu(self.fc_out(h2))
        return q

class mu_net(nn.Module):
    def __init__(self, observation_dim, action_space, l1_channel = 256, l2_channel = 128):
        super(mu_net, self).__init__()
        self.action_space = action_space
        self.fc1 = nn.Linear(observation_dim, l1_channel)
        self.fc2 = nn.Linear(l1_channel, l2_channel)
        self.fc_mu = nn.Linear(l2_channel, action_space.shape[0])

        # f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        # nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        # nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        #
        # f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        # nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        # nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        #
        # f3 = 0.003 # specified in the paper
        # nn.init.uniform_(self.fc_mu.weight.data, -f3, f3)
        # nn.init.uniform_(self.fc_mu.bias.data, -f3, f3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))*max(self.action_space.high)
        return mu