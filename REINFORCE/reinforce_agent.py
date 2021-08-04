import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random

class REINFORCE():
    def __init__(self, input_shape, action_size, seed, device, gamma, lr, policy):
        self.input_shape = input_shape
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.lr = lr
        self.gamma = gamma

        self.policy_net = policy(input_shape, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = self.lr)

        self.log_probs = []
        self.rewards = []
        self.masks = []

    def step(self, log_prob, reward, done):
        self.log_probls.append(log_prob)
        self.rewards.append(torch.from_numpy(np.array[reward]).to(self.devicedevice))
        self.masks.append(torch.from_numpy(np.array([1-done])).to(self.device))

    def act(self, state):
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        action_probs = self.policy_net(state)

        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)

        return action.item(), log_prob

    def learn(self):
        returns = self.compute_returns(0, self.gamma)

        log_probs = torch.cat(self.log_probs)
        returns = torch.cat(returns).detach()

        loss = -(log_probs*returns).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.reset_memory()

    def reset_memory(self):
        del 