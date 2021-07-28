import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

def rargmax(vector):
    m = np.amax(vector) # return max value
    indices = np.nonzero(vector == m)[0] # np.nonzero() returns as (array([3]), )
    return pr.choice(indices) # If all equal, random choice

register (
    id = 'FrozenLake-v3',
    entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name':'4x4',
    'is_slippery':False}
)

env = gym.make('FrozenLake-v3')

Q = np.zeros([env.observation_space.n, env.action_space.n])
num_episodes = 2000
dis = 0.99

rList = []
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False
    e = 1. / ((i//100)+1) # epsilon

    while not done:
        # action = rargmax(Q[state,:])
        # action = np.argmax(Q[state,:] + np.random.randn(1, env.action_space.n) / (i+1)) # random noise for exploration
        if np.random.rand(1) < e: # epsilon-greedy
            action = env.action_space.sample() # random
        else:
            action = np.argmax(Q[state,:]) # greedy

        new_state, reward, done, _ = env.step(action)

        Q[state,action] = reward + dis*np.max(Q[new_state,:])

        rAll += reward
        state = new_state

    rList.append(rAll) # reward 1 if the agent reaches the goal

print(sum(rList))
print(Q)