import sys; sys.path.append('..')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym

import matplotlib.pyplot as plt

from Preprocessing import preprocess_frame, prepare_training_inputs, stack_frame

import numpy as np
from vanilla_DQN import DQN
from vanilla_CNN import CNN
from ReplayBuffer import ReplayBuffer

env = gym.make('PongNoFrameskip-v4')
env.seed(0)

# State and action space check
print('Size of frame: ', env.observation_space.shape)
print('Action space: ', env.action_space.n)

# Visualize Frame
# env.reset()
# plt.figure()
# plt.imshow(env.reset())
# plt.title('Original Frame')
# plt.show()

# Grayscale and Preprocessed Image
# env.reset()
# plt.figure()
# plt.imshow(preprocess_frame(env.reset(), (30, -4, -12, 4), 84), cmap="gray")
# plt.title('Pre Processed image')
# plt.show()

lr = 1e-4 * 5
batch_size = 256
gamma = 1.0
memory_size = 50000
total_eps = 5000
eps_max = 0.080
eps_min = 0.01
sampling_only_until = 2000
target_update_interval = 10

# in_channels = 4 # 4 input images 
# input_shape = 4*84*84

# PATH = './RL_practice/DQN/weights/'

# print(torch.cuda.is_available())

# env = gym.make('CartPole-v1')
# memory = ReplayBuffer(memory_size)

# qnet = CNN(in_channels, env.action_space.n)
# qnet_target = CNN(in_channels, env.action_space.n)

# qnet_target.load_state_dict(qnet.state_dict())
# agent = DQN(input_shape,1,qnet=qnet, qnet_target=qnet_target, lr=lr, gamma=gamma, epsilon = 1.0)

# print_every = 100

# for n_epi in range(total_eps):
#     epsilon = max(eps_min, eps_max-eps_min*(n_epi/200))
#     agent.epsilon = torch.tensor(epsilon)
#     s = np.stack(env.reset()
#     cum_r = 0

#     while True:
#         # env.render()
#         # s = to_tensor(s,size=(1,4))
#         a = agent.get_action(s)
#         ns, r, done, info = env.step(a)

#         experience = (s,torch.tensor(a).view(1,1),torch.tensor(r/100.0).view(1,1), torch.tensor(ns).view(1,4),torch.tensor(done).view(1,1))
#         memory.push(experience)

#         s = ns
#         cum_r += r
#         if done:
#             break

#     if len(memory) >= sampling_only_until:
#         sampled_exps = memory.sample(batch_size)
#         sampled_exps = prepare_training_inputs(sampled_exps)
#         agent.update(*sampled_exps)

#     if n_epi % target_update_interval == 0:
#         qnet_target.load_state_dict(qnet.state_dict())
    
#     if n_epi % print_every == 0:
#         msg = (n_epi, cum_r, epsilon)
#         print("Episode : {:4.0f} | Cumulative Reward : {:4.0f} | Epsilon: {:.3f}".format(*msg))

# torch.save(qnet.state_dict(), PATH + 'Naive_DQN_state_dict.pt')

def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, (30, -4, -12, 4), 84)
    frames = stack_frame(frames, frame, is_new)

    return frames

state = stack_frames(None, env.reset(), True)
print(state.shape)