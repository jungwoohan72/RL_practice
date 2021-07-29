# import sys; sys.path.append('..')

import torch
import gym

import numpy as np

from MLP import MLP as MLP

def to_tensor(np_array:np.array, size=None):
    torch_tensor = torch.from_numpy(np_array).float()
    if size is not None:
        torch_tensor = torch_tensor.view(size)
    return torch_tensor

PATH = './RL_practice/DQN/weights/'

model = MLP(4,2, num_neurons=[128])
model.load_state_dict(torch.load(PATH + 'Naive_DQN_state_dict.pt'))

env = gym.make('CartPole-v1')

for i_episode in range(20):
    s = env.reset()
    cum_r = 0

    while True:
        env.render()
        s = to_tensor(s,size=(1,4))
        a = int(model(s).argmax(dim=-1))

        ns, r, done, info = env.step(a)

        s = ns
        cum_r += r

        if done:
            break

    print("Episode {} done".format(i_episode))
    print("Cumulative Reward : {:4.0f}".format(cum_r))

env.close()
