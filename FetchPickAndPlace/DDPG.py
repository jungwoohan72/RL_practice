import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DDPG(nn.Module):
    def __init__(self, q_net, q_target, mu_net, mu_target, lr_mu, lr_q, gamma, tau):
        super(DDPG, self).__init__()
        self.qnet = q_net
        self.q_target = q_target
        self.munet = mu_net
        self.mu_target = mu_target
        self.lr_mu = lr_mu
        self.lr_q = lr_q
        self.gamma = gamma
        self.tau = tau
        self.q_optim = optim.Adam(self.qnet.parameters(), lr = lr_q)
        self.mu_optim = optim.Adam(self.munet.parameters(), lr = lr_mu)

    def soft_update(self, net, target):
        for param_target, param in zip(target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def train(self, memory, batch_size):
        s,a,r,s_prime,done = memory.sample(batch_size)

        # print(s.shape, a.shape, r.shape, s_prime.shape, done.shape)
        # torch size: (batch_size, state_dim), (batch_size, action_dim), (batch_size, 1), (batch_size, state_dim), (batch_size, 1)

        self.q_optim.zero_grad()
        target = r + self.gamma*self.q_target(s_prime, self.mu_target(s_prime))*(1-done)
        q_loss = (self.qnet(s,a) - target.detach()).pow(2).mean()
        q_loss.backward(inputs=list(self.qnet.parameters())) # I don't know why inplace error occurs here without inputs argument!!!
        self.q_optim.step()

        mu_loss = -self.qnet(s,self.munet(s))
        mu_loss = mu_loss.mean()
        self.mu_optim.zero_grad()
        mu_loss.backward()
        self.mu_optim.step()

        self.soft_update(self.qnet, self.q_target)
        self.soft_update(self.munet, self.mu_target)

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x