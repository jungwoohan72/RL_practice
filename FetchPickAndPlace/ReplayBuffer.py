import torch
import collections
import random

class ReplayBuffer():
    def __init__(self, buffer_limit = 50000, device = 'cpu'):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.device = device

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done])

        states = torch.tensor(s_lst).float().to(self.device)
        actions = torch.tensor(a_lst).float().to(self.device)
        rewards = torch.tensor(r_lst).float().to(self.device)
        next_states = torch.tensor(s_prime_lst).float().to(self.device)
        dones = torch.tensor(done_mask_lst).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)