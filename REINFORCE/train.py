import gym
import torch
import torch.nn as nn
import numpy as np

from torch.distributions.categorical import Categorical
from MLP import MLP as MLP
from REINFORCE import REINFORCE as REINFORCE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA = 0.9
lr = 0.0002
n_eps = 10000
print_every = 500

env = gym.make('CartPole-v1')
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.n

net = MLP(s_dim, a_dim, [128]).to(DEVICE)
agent = REINFORCE(net, GAMMA, lr).to(DEVICE)

for ep in range(n_eps):
    s = env.reset()
    cum_r = 0

    states = []
    actions = []
    rewards = []

    while True:
        a = agent.get_action(s)
        ns, r, done, info = env.step(a.item())

        states.append(s)
        actions.append(a)
        rewards.append(r)

        cum_r += r
        if done:
            break

    if ep%print_every == 0:
        print("Episode: {} || Cumulative Reward: {}".format(ep, cum_r))


    print(states.shape)
    print(actions.shape)
    print(rewards.shape)
    states = torch.cat(states, dim = 0)
    rewards = torch.tensor(rewards)

    actions = torch.stack(actions).squeeze()
    episode = (states, actions, rewards)
    agent.update_episode()

