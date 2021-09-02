import sys; sys.path.append('..')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym
import math
import json

import matplotlib.pyplot as plt

from Preprocessing import preprocess_image, prepare_training_inputs, stack_frame

import numpy as np
from vanilla_DQN import DQN
from vanilla_CNN import CNN
from ReplayBuffer import ReplayBuffer

import wandb

env = gym.make('PongDeterministic-v4')

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
print(DEVICE)
PATH = './weights/Pong_Deterministic/Pong_DQN_'

SAVE_MODELS = False  # Save models to file so you can test later
MODEL_PATH = './RL_practice/DQN/weights/'  # Models path for saving or loading
SAVE_MODEL_INTERVAL = 30  # Save models at every X epoch
TRAIN_MODEL = True  # Train model while playing (Make it False when testing a model)
TARGET_UPDATE_INTERVAL = 10
TRAIN_FLAG = True

LOAD_MODEL_FROM_FILE = False  # Load model from file
LOAD_FILE_EPISODE = 510  # Load Xth episode from file

BATCH_SIZE = 64  # Minibatch size that select randomly from mem for train nets
MAX_EPISODE = 100000  # Max episode
MAX_STEP = 100000  # Max step size for one episode

MAX_MEMORY_LEN = 50000  # Max memory len
MIN_MEMORY_LEN = 40000  # Min memory len before start train

GAMMA = 0.97  # Discount rate
ALPHA = 1e-3  # Learning rate
EPS_MIN = 0.010
EPS_MAX = 1
EPS_DECAY = 0.99

RENDER_GAME_WINDOW = False  # Opens a new window to render the game (Won't work on colab default)

IN_CHANNELS = 4 # 4 sequential frames as input -> 4 channel input

memory = ReplayBuffer(MAX_MEMORY_LEN)

qnet = CNN(IN_CHANNELS, env.action_space.n).to(DEVICE)
qnet_target = CNN(IN_CHANNELS, env.action_space.n).to(DEVICE)
qnet_target.load_state_dict(qnet.state_dict())

agent = DQN(4,env.action_space.n,qnet=qnet, qnet_target=qnet_target, lr=ALPHA, gamma=GAMMA, epsilon = 1.0).to(DEVICE) # 4 sequential channel input

print_every = 1

config = dict(
    learning_rate = ALPHA,
    gamma = GAMMA
)
wandb.init(project='pong-DQN',
        config = config)

if LOAD_MODEL_FROM_FILE:
    agent.qnet.load_state_dict(torch.load(PATH+str(LOAD_FILE_EPISODE)+".pkl"))
    agent.qnet_target.load_state_dict(qnet.state_dict())

    with open(PATH+str(LOAD_FILE_EPISODE)+'.json') as outfile:
        param = json.load(outfile)
        agent.epsilon = torch.tensor(param.get('epsilon')).to(DEVICE)

    startEpisode = LOAD_FILE_EPISODE + 1

else:
    startEpisode = 0

for n_epi in range(startEpisode, MAX_EPISODE):
    agent.random_action = 0
    agent.greedy_action = 0

    # epsilon = max(EPS_MIN, EPS_MAX - EPS_MIN * (n_epi / 200))
    # epsilon = EPS_MIN + (EPS_MAX - EPS_MIN) * math.exp(-1. * n_epi / 1000)
    epsilon = max(EPS_MIN, EPS_MAX*(EPS_DECAY**n_epi))
    agent.epsilon = torch.tensor(epsilon).to(DEVICE)

    state = env.reset()
    state = preprocess_image(state, (20, env.observation_space.shape[0], 0, env.observation_space.shape[1]), 84, 64)

    s = np.stack((state, state, state, state))
    s = torch.from_numpy(s).float().unsqueeze(0).to(DEVICE)

    cum_r = 0

    for step in range(MAX_STEP):
        if RENDER_GAME_WINDOW:
            env.render()

        a = agent.get_action(s, DEVICE)

        next_state, r, done, info = env.step(a)
        next_state = preprocess_image(next_state, (20, env.observation_space.shape[0], 0, env.observation_space.shape[1]), 84, 64)
        next_state = torch.from_numpy(next_state).float().to(DEVICE)

        temp = s.squeeze()
        ns = torch.stack((next_state, temp[0,:,:], temp[1,:,:], temp[2,:,:]), dim = 0).unsqueeze(0)

        if TRAIN_MODEL:
            experience = (s, torch.tensor(a).view(1,1).to(DEVICE), torch.tensor(r).view(1,1).to(DEVICE), ns, torch.tensor(done).view(1,1).to(DEVICE))
            memory.push(experience)

        s = ns
        cum_r += r

        if len(memory) >= MIN_MEMORY_LEN:
            if TRAIN_FLAG == True:
                print("Training Start")
                TRAIN_FLAG = False
            sampled_exps = memory.sample(BATCH_SIZE)
            sampled_exps = prepare_training_inputs(sampled_exps, device = DEVICE)
            agent.update(*sampled_exps)

        if done:
            epsilonDict = {'epsilon': epsilon}  # Create epsilon dict to save model as file
            if SAVE_MODELS and n_epi % SAVE_MODEL_INTERVAL == 0:  # Save model as file
                weightsPath = PATH + str(n_epi) + '.pkl'
                epsilonPath = PATH + str(n_epi) + '.json'

                torch.save(agent.qnet.state_dict(), weightsPath)
                with open(epsilonPath, 'w') as outfile:
                    json.dump(epsilonDict, outfile)
            break

    if n_epi % TARGET_UPDATE_INTERVAL == 0:
        print("Target Network Updated!")
        qnet_target.load_state_dict(qnet.state_dict())

    if n_epi % print_every == 0:
        msg = (n_epi, cum_r, epsilon, agent.loss, agent.random_action, agent.greedy_action)
        print("Episode : {:4.0f} | Cumulative Reward : {:4.0f} | Epsilon: {:.3f} | Loss: {:.6f} | Random: {} | Greedy: {}".format(*msg))

    log_dict = dict()
    log_dict['cum_return'] = cum_r
    log_dict['epsilon'] = epsilon
    wandb.log(log_dict)

# Save model and experiment configuration
json_val = json.dumps(config)
with open(join(wandb.run.dir, 'config.json'), 'w') as f:
    json.dump(json_val, f)

torch.save(agent.qnet.state_dict(), join(wandb.run.dir, 'model.pt'))

# close wandb session
wandb.join()