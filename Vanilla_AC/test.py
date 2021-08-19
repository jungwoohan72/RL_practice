import gym
import torch
import torch.nn as nn
import numpy as np

from torch.distributions.categorical import Categorical
from MLP import MLP as MLP
from vanilla_AC import A2C as A2C
from REINFORCE import REINFORCE as REINFORCE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

n_eps = 100
print_every = 1

env = gym.make('CartPole-v1')
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.n

policy_net = MLP(s_dim, a_dim, [128]).to(DEVICE)

policy_net.load_state_dict(torch.load('./weights/Cartpole_A2C.pkl'))

for ep in range(n_eps):
    s = env.reset()
    cum_r = 0

    while True:
        env.render()
        s = torch.from_numpy(s).float().to(DEVICE) # torch.Size([4])
        s = s.view((1, 4)) # torch.Size([1,4])
        a = torch.argmax(policy_net(s))

        ns, r, done, info = env.step(a.item())

        s = ns

        cum_r += r
        if done:
            break

    if ep % print_every == 0:
        print("Episode: {} || Cumulative Reward: {}".format(ep, cum_r))
