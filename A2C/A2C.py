import torch
import torch.nn as nn

from torch.distributions.categorical import Categorical

class A2C(nn.Module):
    def __init__(self, policy_net, value_net, gamma, lr, device):
        super(A2C, self).__init__()
        self.policy_net = policy_net
        self.value_net = value_net
        self.gamma = gamma
        self.lr = lr

        params = list(policy_net.parameters()) + list(value_net.parameters())
        self.optimizer = torch.optim.Adam(params = params, lr = lr)

        self._eps = 1e-25
        self._mse = torch.nn.MSELoss()
        self.device = device

    def get_action(self, state):
        with torch.no_grad():
            logits = self.policy_net(state).to(self.device)
            dist = Categorical(logits = logits)
            a = dist.sample() # torch.Size([1])
        return a

    def update(self, state, action, reward, next_state, done):

        # action size: torch.Size([1,1])

        next_state = torch.from_numpy(next_state).float().to(self.device)
        next_state = next_state.view((1,4)) # value_net input은 size [1,4]여야 함.
        reward = torch.tensor(reward).to(self.device)

        with torch.no_grad():
            td_target = reward + self.gamma * self.value_net(next_state) * (1-done)
            td_error = td_target - self.value_net(state)

        dist = Categorical(logits = self.policy_net(state)) # torch.Size([1,2])
        prob = dist.probs.gather(1,action)

        v = self.value_net(state)

        loss = -torch.log(prob + self._eps)*td_error + self._mse(v, td_target*td_error)  # policy loss + value loss / shape: torch.Size([1,1])
        loss = loss.mean() # shape: torch.Size([])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()