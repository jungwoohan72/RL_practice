import gym
import torch

from MLP import MLP as MLP
from REINFORCE import REINFORCE as REINFORCE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
GAMMA = 0.9
lr = 0.0002
n_eps = 10
print_every = 1

env = gym.make('CartPole-v1')
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.n

net = MLP(s_dim, a_dim, [128]).to(DEVICE)
agent = REINFORCE(net, GAMMA, lr, DEVICE).to(DEVICE)

agent.policy.load_state_dict(torch.load('./weights/Cartpole_REINFORCE.pkl'))

for ep in range(n_eps):
    s = env.reset()
    cum_r = 0

    while True:
        env.render()
        s = torch.from_numpy(s).float().to(DEVICE)
        s = s.view((1, 4))
        a = agent.get_action(s)
        ns, r, done, info = env.step(a.item())

        s = ns
        cum_r += r
        if done:
            break
    if ep % print_every == 0:
        print("Episode: {} || Cumulative Reward: {}".format(ep, cum_r))
