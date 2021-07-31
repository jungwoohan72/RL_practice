# import sys; sys.path.append('..')

import torch
import gym

import numpy as np

from Preprocessing import preprocess_image, prepare_training_inputs, stack_frame
from vanilla_CNN import CNN

from MLP import MLP as MLP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH = './weights/'

SAVE_MODELS = True  # Save models to file so you can test later
MODEL_PATH = './RL_practice/DQN/weights/'  # Models path for saving or loading
SAVE_MODEL_INTERVAL = 10  # Save models at every X epoch
TRAIN_MODEL = True  # Train model while playing (Make it False when testing a model)
TARGET_UPDATE_INTERVAL = 10
TRAIN_FLAG = True

LOAD_MODEL_FROM_FILE = True  # Load model from file
LOAD_FILE_EPISODE = 0  # Load Xth episode from file

BATCH_SIZE = 64  # Minibatch size that select randomly from mem for train nets
MAX_EPISODE = 1000  # Max episode
MAX_STEP = 100000  # Max step size for one episode

MAX_MEMORY_LEN = 50000  # Max memory len
MIN_MEMORY_LEN = 40000  # Min memory len before start train

GAMMA = 0.97  # Discount rate
ALPHA = 1e-4 * 5  # Learning rate
EPS_MIN = 0.010
EPS_MAX = 0.99

RENDER_GAME_WINDOW = True  # Opens a new window to render the game (Won't work on colab default)

IN_CHANNELS = 4 # 4 sequential frames as input -> 4 channel input

env = gym.make('PongNoFrameskip-v4')
model = CNN(IN_CHANNELS, env.action_space.n).to(DEVICE)
model.load_state_dict(torch.load(PATH + 'Pong_DQN.pt'))

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
