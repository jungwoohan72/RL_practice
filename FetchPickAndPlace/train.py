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
from DDPG import DDPG as DDPG, OrnsteinUhlenbeckNoise

def main():

    batch_size = 32
    lr_mu = 0.0005
    lr_q = 0.001
    gamma = 0.99
    tau = 0.005
    print_interval = 100

    env = gym.make('BipedalWalkerHardcore-v3')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]

    print("Observation dimension:", s_dim, "\nAction dimension: ", a_dim)
    print("Action range: -1 to 1")

    memory = ReplayBuffer(buffer_limit=50000, device = device)

    q, q_target = Qnet(s_dim, a_dim).to(device), Qnet(s_dim, a_dim).to(device)
    mu, mu_target = mu_net(s_dim, a_dim).to(device), mu_net(s_dim, a_dim).to(device)

    q_target.load_state_dict(q.state_dict())
    mu_target.load_state_dict(mu.state_dict())

    agent = DDPG(q, q_target, mu, mu_target, lr_mu, lr_q, gamma, tau)
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(4))

    for n_epi in range(10000):
        s = env.reset()
        done = False

        score = 0.0

        while not done:
            s = torch.from_numpy(s).float().to(device)
            a = agent.munet(s)

            s_prime, r, done, info = env.step(a.cpu().detach().numpy()+ou_noise())
            memory.put((s.unsqueeze(0), a.unsqueeze(0), torch.tensor(r/100.0).view(1,1), torch.from_numpy(s_prime).float().unsqueeze(0), torch.tensor(done).view(1,1)))
            score += r
            s = s_prime

        if memory.size()>2000:
            for i in range(10):
                with torch.autograd.set_detect_anomaly(True):
                    agent.train(memory, batch_size)
                    print("train")
                agent.soft_update(agent.qnet, agent.q_target)
                agent.soft_update(agent.munet, agent.mu_target)

        if n_epi % print_interval == 0:
            print("Episode: {}, Score: {:.1f}".format(n_epi, score))

    env.close()

if __name__ == '__main__':
    main()