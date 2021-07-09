import numpy as np
import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, state_dim:int, action_dim:int, qnet:nn.Module, qnet_target:nn.Module, lr:float, gamma:float, epsilon: float):
        super(DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.qnet = qnet
        self.lr = lr
        self.gamma = gamma
        self.opt = torch.optim.Adam(params=self.qnet.parameters(), lr=lr)
        self.register_buffer('epsilon',torch.ones(1))

        self.qnet_target = qnet_target
        self.criteria = nn.SmoothL1Loss()

    def get_action(self, state):
        qs = self.qnet(state) # output possible actions (+1 or -1) at certain state
        prob = np.random.uniform(0.0, 1.0, 1) 

        # epsilon-greedy
        if torch.from_numpy(prob).float() <= self.epsilon: # random
            action = np.random.choice(range(self.action_dim)) # choose random action which has dimension of self.action_dim
        else:
            action = qs.argmax(dim=-1)
        return int(action)

    def update(self, state, action,reward, next_state, done):
        s, a, r, ns = state, action, reward, next_state # provided by step method of env

        with torch.no_grad():
            q_max, _ = self.qnet_target(ns).max(dim=-1, keepdims = True) # greedy policy for q_target
            q_target = r + self.gamma*q_max*(1-done)

        q_val = self.qnet(s).gather(1,a)
        loss = self.criteria(q_val, q_target)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()