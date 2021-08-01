import torch
import gym

import numpy as np

from Preprocessing import preprocess_image
from vanilla_CNN import CNN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH = './weights/Pong_Deterministic/Pong_DQN_'

RENDER_GAME_WINDOW = True  # Opens a new window to render the game (Won't work on colab default)

IN_CHANNELS = 4 # 4 sequential frames as input -> 4 channel input

env = gym.make('PongDeterministic-v4')
model = CNN(IN_CHANNELS, env.action_space.n).to(DEVICE)
model.load_state_dict(torch.load(PATH + '2430.pkl'))

print_every = 10

for n_epi in range(100):
    state = env.reset()
    state = preprocess_image(state, (20, env.observation_space.shape[0], 0, env.observation_space.shape[1]), 84, 64)

    s = np.stack((state, state, state, state))
    s = torch.from_numpy(s).float().unsqueeze(0).to(DEVICE)

    cum_r = 0

    while True:
        if RENDER_GAME_WINDOW:
            env.render()

        a = int(model(s).argmax(dim=-1))
        next_state, r, done, info = env.step(a)
        next_state = preprocess_image(next_state, (20, env.observation_space.shape[0], 0, env.observation_space.shape[1]), 84, 64)
        next_state = torch.from_numpy(next_state).float().to(DEVICE)

        temp = s.squeeze()
        ns = torch.stack((next_state, temp[0,:,:], temp[1,:,:], temp[2,:,:]), dim = 0).unsqueeze(0)

        s = ns
        cum_r += r
        if done:
            break

    msg = (n_epi, cum_r)
    print("Episode : {:4.0f} | Cumulative Reward : {:4.0f}".format(*msg))
