import torch
import torch.nn as nn
import torch.nn.functional as F

# 보통 Q-function을 parameterize할 때 observation을 input으로 받고 output을 action space dimension에 맞게 한다.
# 하지만 continuous space의 deterministic action은 그게 불가능하다.

class Qnet(nn.Module):
    def __init__(self, observation_channel):
        super(Qnet, self).__init__()
        self.fc_s = nn.Linear(observation_channel, 64)
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
