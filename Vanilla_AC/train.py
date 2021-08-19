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
GAMMA = 1 # very important
lr = 0.0002
n_eps = 10000
print_every = 10

env = gym.make('CartPole-v1')
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.n

policy_net = MLP(s_dim, a_dim, [128]).to(DEVICE)
value_net = MLP(s_dim, 1, [128]).to(DEVICE)
agent = A2C(policy_net, value_net, GAMMA, lr, DEVICE).to(DEVICE)

for ep in range(n_eps):
    s = env.reset()
    cum_r = 0

    while True:
        # env.render()
        s = torch.from_numpy(s).float().to(DEVICE) # torch.Size([4])
        s = s.view((1, 4)) # torch.Size([1,4])
        a = agent.get_action(s)

        ns, r, done, info = env.step(a.item())

        agent.update(s, a.view(-1,1), r, ns, done)

        s = ns

        cum_r += r
        if done:
            break

    if ep % print_every == 0:
        print("Episode: {} || Cumulative Reward: {}".format(ep, cum_r))

torch.save(agent.policy_net.state_dict(), './weights/Cartpole_A2C.pkl')