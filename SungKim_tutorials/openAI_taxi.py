import gym
env = gym.make("Taxi-v3")
observation = env.reset()

for _ in range(100):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

