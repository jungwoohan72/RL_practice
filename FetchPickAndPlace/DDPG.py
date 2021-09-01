import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DDPG(nn.Module):
    def __init__(self, q_net, mu_net, observation_channel, action_channel, lr_mu, lr_q, gamma, device):
        super(DDPG, self).__init__()
        self.qnet = q_net
        self.munet = mu_net
        self.lr_mu = lr_mu
        self.lr_q = lr_q

    def train(self, memory):
        s,a,r,s_prime,done = memory.sample(batch_size)

