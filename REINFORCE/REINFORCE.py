import torch
import torch.nn as nn
import numpy as np

from torch.distributions.categorical import Categorical

class REINFORCE(nn.Module):
    def __init__(self, policy, gamma, lr, device):
        super(REINFORCE, self).__init__()
        self.policy = policy
        self.gamma = gamma
        self.opt = torch.optim.Adam(params = self.policy.parameters(), lr = lr)
        self._eps = 1e-25
        self.device = device

    def get_action(self, state):
        with torch.no_grad():
            logits = self.policy(state).to(self.device)
            dist = Categorical(logits=logits)
            a = dist.sample()
        return a

    def update_step_by_step(self, episode):
        states = episode[0].flip(dims = [0]).to(self.device)
        actions = episode[1].flip(dims = [0]).to(self.device)
        rewards = episode[2].flip(dims = [0]).to(self.device)

        g = 0
        for s,a,r in zip(states, actions, rewards):
            g = r + g*self.gamma
            dist = Categorical(logits = self.policy(s)) # sampling
            prob = dist.probs[a]

            loss = -torch.log(prob + self._eps)*g

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

    def update_episode(self, episode):
        states = episode[0].flip(dims = [0]).to(self.device)
        actions = episode[1].flip(dims = [0]).to(self.device)
        rewards = episode[2].flip(dims = [0]).to(self.device)

        g = 0
        returns = []
        for s,a,r in zip(states, actions, rewards):
            g = r + self.gamma*g
            returns.append(g)

        returns = torch.tensor(returns).to(self.device)

        returns = (returns - returns.mean()) / (returns.std() + self._eps)
        returns.to(self.device)

        dist = Categorical(logits = self.policy(states)) # probability for each action -> sampling
        prob = dist.probs[range(states.shape[0]), actions] # (states.shape[0], 1) tensor

        self.opt.zero_grad()
        loss = -torch.log(prob+self._eps)*returns
        loss = loss.mean()
        loss.backward()
        self.opt.step()

