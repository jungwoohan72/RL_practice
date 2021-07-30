import sys; sys.path.append('..')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym

import matplotlib.pyplot as plt

from Preprocessing import preprocess_image, prepare_training_inputs, stack_frame

import numpy as np
from vanilla_DQN import DQN
from vanilla_CNN import CNN
from ReplayBuffer import ReplayBuffer

env = gym.make('PongNoFrameskip-v4')
env.seed(0)

# State and action space check
print('Size of frame: ', env.observation_space.shape)
print('Action space: ', env.action_space.n)

# Grayscale and Preprocessed Image
# plt.figure()
# plt.imshow(env.reset())
# plt.imshow(preprocess_image(env.reset(), (20, env.observation_space.shape[0], 0, env.observation_space.shape[1]), 84, 64 ), cmap="gray")
# plt.title('Pre Processed image')
# plt.show()

############################################## Main Code ##############################################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH = './RL_practice/DQN/weights/'

SAVE_MODELS = True  # Save models to file so you can test later
MODEL_PATH = './RL_practice/DQN/weights/'  # Models path for saving or loading
SAVE_MODEL_INTERVAL = 10  # Save models at every X epoch
TRAIN_MODEL = True  # Train model while playing (Make it False when testing a model)
TARGET_UPDATE_INTERVAL = 10

LOAD_MODEL_FROM_FILE = False  # Load model from file
LOAD_FILE_EPISODE = 0  # Load Xth episode from file

BATCH_SIZE = 64  # Minibatch size that select randomly from mem for train nets
MAX_EPISODE = 100000  # Max episode
MAX_STEP = 100000  # Max step size for one episode

MAX_MEMORY_LEN = 50000  # Max memory len
MIN_MEMORY_LEN = 10000  # Min memory len before start train

GAMMA = 0.97  # Discount rate
ALPHA = 1e-4 * 5  # Learning rate
EPS_MIN = 0.010
EPS_MAX = 0.080

RENDER_GAME_WINDOW = False  # Opens a new window to render the game (Won't work on colab default)

IN_CHANNELS = 4 # 4 sequential frames as input -> 4 channel input

memory = ReplayBuffer(MAX_MEMORY_LEN)

qnet = CNN(IN_CHANNELS, env.action_space.n)
qnet_target = CNN(IN_CHANNELS, env.action_space.n)
qnet_target.load_state_dict(qnet.state_dict())

agent = DQN(4,1,qnet=qnet, qnet_target=qnet_target, lr=ALPHA, gamma=GAMMA, epsilon = 1.0) # 4 sequential channel input

print_every = 10

for n_epi in range(MAX_EPISODE):
    epsilon = max(EPS_MIN, EPS_MAX-EPS_MIN*(n_epi/200))
    agent.epsilon = torch.tensor(epsilon)
    state = env.reset()
    state = preprocess_image(state, (20, env.observation_space.shape[0], 0, env.observation_space.shape[1]), 84, 64)

    s = np.stack((state, state, state, state))
    s = torch.from_numpy(s).float().unsqueeze(0)

    cum_r = 0

    while True:
        if RENDER_GAME_WINDOW:
            env.render()

        a = agent.get_action(s)
        next_state, r, done, info = env.step(a)
        next_state = preprocess_image(next_state, (20, env.observation_space.shape[0], 0, env.observation_space.shape[1]), 84, 64)
        next_state = torch.from_numpy(next_state).float()

        temp = s.squeeze()
        ns = torch.stack((next_state, temp[0,:,:], temp[1,:,:], temp[2,:,:]), dim = 0).unsqueeze(0)

        if TRAIN_MODEL:
            experience = (s, torch.tensor(a).view(1,1), torch.tensor(r/100.0).view(1,1), ns, torch.tensor(done).view(1,1))
            memory.push(experience)

        s = ns
        cum_r += r
        if done:
            break

    if len(memory) >= MIN_MEMORY_LEN:
        sampled_exps = memory.sample(BATCH_SIZE)
        sampled_exps = prepare_training_inputs(sampled_exps)
        agent.update(*sampled_exps)

    if n_epi % TARGET_UPDATE_INTERVAL == 0:
        qnet_target.load_state_dict(qnet.state_dict())

    if n_epi % print_every == 0:
        msg = (n_epi, cum_r, epsilon)
        print("Episode : {:4.0f} | Cumulative Reward : {:4.0f} | Epsilon: {:.3f}".format(*msg))

if SAVE_MODELS:
    torch.save(qnet.state_dict(), PATH + 'Pong_DQN.pt')
