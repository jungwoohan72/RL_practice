import torch
import torch.nn as nn

from torch.distributions.categorical import Categorical

class A2C(nn.Module):
    def __init__(self, policy_net, value_net, gamma: float = 1.0, lr: float = 0.0002, device):
        super(A2C, self).__init__()
        self.policy_net = policy_net
        self.value_net = value_net
        self.gamma = gamma
        self.lr = lr

        params = list(policy_net.parameters()) + list(value_net.parameters)
        self.optimizer = torch.optim.Adam(params = params, lr = lr)

        self._eps = 1e-25
        self._mse = torch.nn.MSELoss()

    def get_action(self, state):
        with torch.no_grad():
            logits = self.policy_net(state)
            dist = Categorical(logits = logits)
            a = dist.sample()

        return a

    def update(self, state, action, reward, next_state, done):
        with torch.no_grad():
            td_target = reward + self.gamma * self.value_net(next_state) * (1-done)
            td_error = td_target - self.value_net(state)

        dist = Categorical(self.policy_net(state))
        prob = dist.probs.gather(1,action)

        v = self.value_net(state)

        loss = self._mse(v, td_target)
