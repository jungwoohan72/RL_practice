import sys; sys.path.append('..')

import gym
import random
import collections
import wandb
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MuNet(nn.Module):
    def __init__(self, observation_dim, action_space, l1_channel = 256, l2_channel = 128):
        super(MuNet, self).__init__()
        self.action_space = action_space
        self.fc1 = nn.Linear(observation_dim, l1_channel)
        self.fc2 = nn.Linear(l1_channel, l2_channel)
        self.fc_mu = nn.Linear(l2_channel, action_space.shape[0])

        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        f3 = 0.003 # specified in the paper
        nn.init.uniform_(self.fc_mu.weight.data, -f3, f3)
        nn.init.uniform_(self.fc_mu.bias.data, -f3, f3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))*max(self.action_space.high)
        return mu

def main():
    env = gym.make('Pendulum-v0')
    # env = gym.make('BipedalWalker-v3')
    # env = gym.make('LunarLanderContinuous-v2')

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    print("State dim: ", s_dim, "Action dim: ", a_dim)

    mu = MuNet(s_dim, env.action_space, l1_channel = 400, l2_channel = 300)

    mu.load_state_dict(torch.load(PATH))

    for n_epi in range(10):
        s = env.reset()
        ep_score = 0
        done = False
        while not done:
            env.render()
            a = mu(torch.from_numpy(s).float())
            a = a.detach().numpy()
            s_prime, r, done, info = env.step(a)

            ep_score += r
            s = s_prime

        print("# of episode :{}, score : {:.1f}".format(n_epi, ep_score))

    env.close()

if __name__ == '__main__':

    PATH = './pendulum/files_mu_1000.pt'

    main()

