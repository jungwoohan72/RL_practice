import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import Qnet, mu_net
from ReplayBuffer import ReplayBuffer as ReplayBuffer

def main():

    env = gym.make('BipedalWalkerHardcore-v3')

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]

    print("Observation dimension:", s_dim, "\nAction dimension: ", a_dim)
    print("Action range: -1 to 1")

    memory = ReplayBuffer(buffer_limit=50000)

    q, q_target = MLP()

if __name__ == '__main__':
    main()