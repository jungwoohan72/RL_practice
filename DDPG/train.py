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

class ReplayBuffer():
    def __init__(self, device = 'cpu'):
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
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float).to(self.device), torch.tensor(a_lst, dtype=torch.float).to(self.device), \
               torch.tensor(r_lst, dtype=torch.float).to(self.device), torch.tensor(s_prime_lst, dtype=torch.float).to(self.device), \
               torch.tensor(done_mask_lst, dtype=torch.float).to(self.device)

    def size(self):
        return len(self.buffer)

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

class QNet(nn.Module):
    def __init__(self, observation_dim, action_space, l1_channel = 256, l2_channel = 128):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(observation_dim, int(l1_channel/2))
        self.fc_a = nn.Linear(action_space.shape[0],int(l1_channel/2))
        self.fc_q = nn.Linear(l1_channel, l2_channel)
        self.fc_out = nn.Linear(l2_channel,1)

        f1 = 1./np.sqrt(self.fc_s.weight.data.size()[0])
        nn.init.uniform_(self.fc_s.weight.data, -f1, f1)
        nn.init.uniform_(self.fc_s.bias.data, -f1, f1)

        f2 = 1./np.sqrt(self.fc_a.weight.data.size()[0])
        nn.init.uniform_(self.fc_a.weight.data, -f2, f2)
        nn.init.uniform_(self.fc_a.bias.data, -f2, f2)

        f3 = 0.003 # specified in the paper
        nn.init.uniform_(self.fc_out.weight.data, -f3, f3)
        nn.init.uniform_(self.fc_out.bias.data, -f3, f3)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.15, 0.01, 0.2
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
    s,a,r,s_prime,done_mask  = memory.sample(batch_size)

    # print(s.shape, a.shape, r.shape,s_prime.shape, done_mask.shape)
    with torch.no_grad():
        target = r + gamma * q_target(s_prime, mu_target(s_prime)) * done_mask
    q_loss = F.mse_loss(q(s,a), target.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()

    mu_loss = -q(s,mu(s)).mean() # That's all for the policy loss.
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()

def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

def main():

    if config["env"] == "Pendulum":
        env = gym.make('Pendulum-v0')
        max_ep_length = 1000
        render_flag = 100
        save_flag = 1000
    elif config["env"] == "BipedalWalker":
        max_ep_length = 25000
        render_flag = 1000
        save_flag = 1000
        env = gym.make('BipedalWalker-v3')
    elif config["env"] == "MountainCar":
        max_ep_length = 200
        render_flag = 20
        save_flag = 100
        env = gym.make('MountainCarContinuous-v0')
    elif config["env"] == "LunarLander":
        max_ep_length = 2000
        render_flag = 100
        save_flag = 500
        env = gym.make('LunarLanderContinuous-v2')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    print("State dim: ", s_dim, "Action dim: ", a_dim)

    memory = ReplayBuffer(device = device)

    q, q_target = QNet(s_dim, env.action_space, l1_channel = 400, l2_channel = 300).to(device), QNet(s_dim, env.action_space, l1_channel = 400, l2_channel = 300).to(device)
    q_target.load_state_dict(q.state_dict())
    mu, mu_target = MuNet(s_dim, env.action_space, l1_channel = 400, l2_channel = 300).to(device), MuNet(s_dim, env.action_space, l1_channel = 400, l2_channel = 300).to(device)
    mu_target.load_state_dict(mu.state_dict())

    score = 0.0
    print_interval = 20
    train_flag = True

    log_dict = dict()

    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    q_optimizer  = optim.Adam(q.parameters(), lr=lr_q, weight_decay = 0.01)
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(env.action_space.shape[0]))

    for n_epi in range(max_ep_length):
        s = env.reset()
        ep_score = 0

        done = False
        step = 0
        while not done and step < env._max_episode_steps:
            step += 1
            if n_epi % render_flag == 0:
                env.render()
            a = mu(torch.from_numpy(s).float().to(device))
            a = a.cpu().detach().numpy() + ou_noise()
            s_prime, r, done, info = env.step(a)
            memory.put((s,a,r,s_prime,done))
            score += r
            ep_score += r
            s = s_prime

            if memory.size()>2000:
                if train_flag == True:
                    train_flag = False
                    print("Train Starts!")
                train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
                soft_update(mu, mu_target)
                soft_update(q,  q_target)

        if n_epi%print_interval==0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            log_dict['20_epi_avg_return'] = score/print_interval
            score = 0.0

        if n_epi % save_flag == 0:
            torch.save(q.state_dict(), ''.join((wandb.run.dir, '_model_' + str(n_epi) + '.pt')))
            torch.save(q_target.state_dict(), ''.join((wandb.run.dir, '_target_' + str(n_epi) + '.pt')))
            torch.save(mu.state_dict(), ''.join((wandb.run.dir, '_mu_' + str(n_epi) + '.pt')))
            torch.save(mu_target.state_dict(), ''.join((wandb.run.dir, '_mu_target_' + str(n_epi) + '.pt')))

        log_dict['ep_return'] = ep_score
        wandb.log(log_dict)

    torch.save(mu.state_dict(), ''.join((wandb.run.dir, '_mu_final.pt')))
    env.close()

if __name__ == '__main__':

    # Hyperparameters
    lr_mu = 0.0001
    lr_q = 0.001
    gamma = 0.99
    batch_size = 64
    buffer_limit = 1000000
    tau = 0.001  # for target network soft update

    # env list: Pendulum BipedalWalker MountainCar LunarLander
    config = {
        "env":"MountainCarContinuous",
        "q_lr":lr_q,
        "mu_lr":lr_mu,
        "gamma":gamma,
        "tau":tau,
        "batch_size":batch_size
    }

    wandb.init(project='MountainCarContinuous-DDPG',
               config=config)

    main()

    json_val = json.dumps(config)
    with open(''.join((wandb.run.dir, 'config.json')), 'w') as f:
        json.dump(json_val, f)

