import gym
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

import numpy as np

from MLP import MLP as MLP

def to_tensor(np_array:np.array, size=None):
    torch_tensor = torch.from_numpy(np_array).float()
    if size is not None:
        torch_tensor = torch_tensor.view(size)
    return torch_tensor

class EMAMeter:

    def __init__(self,
                 alpha: float = 0.5):
        self.s = None
        self.alpha = alpha

    def update(self, y):
        if self.s is None:
            self.s = y
        else:
            self.s = self.alpha * y + (1 - self.alpha) * self.s

class REINFORCE(nn.Module):
    def __init__(self, policy:nn.Module, gamma: float=1.0, lr: float = 0.0002):
        super(REINFORCE, self).__init__()
        self.policy = policy
        self.gamma = gamma
        self.opt = torch.optim.Adam(params = self.policy.parameters(),lr = lr)
        self._eps = 1e-25

    def get_action(self, state):
        with torch.no_grad():
            logits = self.policy(state)
            dist = Categorical(logits=logits)
            print(dist)
            a = dist.sample()
            print(a)
        return a

    @staticmethod
    def _pre_process_inputs(episode):
        states, actions, rewards = episode

        # assume inputs as follows
        # s : torch.tensor [num.steps x state_dim]
        # a : torch.tensor [num.steps]
        # r : torch.tensor [num.steps]

        # reversing inputs
        states = states.flip(dims=[0])
        actions = actions.flip(dims=[0])
        rewards = rewards.flip(dims=[0])
        return states, actions, rewards

    def update(self, episode):
        # sample-by-sample update version of REINFORCE
        # sample-by-sample update version is highly inefficient in computation
        states, actions, rewards = self._pre_process_inputs(episode)

        g = 0
        for s, a, r in zip(states, actions, rewards):
            g = r + self.gamma * gmodule
            self.opt.zero_grad()

            pg_loss.backward()
            self.opt.step()

env = gym.make('CartPole-v1')
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.n

net = MLP(s_dim, a_dim, [128])
agent = REINFORCE(net)
ema = EMAMeter()

n_eps = 10000
print_every = 500

for ep in range(n_eps):
    s = env.reset()
    cum_r = 0

    states = []
    actions = []
    rewards = []

    while True:
        s = to_tensor(s, size=(1, 4))
        a = agent.get_action(s)
        ns, r, done, info = env.step(a.item())

        states.append(s)
        actions.append(a)
        rewards.append(r)

        s = ns
        cum_r += r
        if done:
            break

    ema.update(cum_r)
    if ep % print_every == 0:
        print("Episode {} || EMA: {} ".format(ep, ema.s))

    states = torch.cat(states, dim=0)  # torch.tensor [num. steps x state dim]
    actions = torch.stack(actions).squeeze()  # torch.tensor [num. steps]
    rewards = torch.tensor(rewards)  # torch.tensor [num. steps]

    episode = (states, actions, rewards)
    agent.update_episode(episode, use_norm=True)