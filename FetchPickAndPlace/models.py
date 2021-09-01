import torch
import torch.nn as nn
import torch.nn.functional as F

# 보통 Q-function을 parameterize할 때 observation을 input으로 받고 output을 action space dimension에 맞게 한다.
# 하지만 continuous space의 deterministic action은 그게 불가능하다.

class Qnet(nn.Module):
    def __init__(self, observation_dim):
        super(Qnet, self).__init__()
        self.fc_s = nn.Linear(dim, 64)
        self.fc_a = nn.Linear(1, 64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim = 1)
        q = F.relu(self.q(cat))
        q = self.fc_out(q)
        return q

class mu_net(nn.Module):
    def __init__(self, observation_dim):
        super(mu_net, self).__init__()
        self.fc1 = nn.Linear(observation_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))
        return mu
