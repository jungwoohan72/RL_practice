def main():
    env = gym.make('CartPole-v1')
    model = PPO()
    gamma = 0.99
    T = 20 # 20 step만큼 경험 쌓고, policy 업데이트 -> On-policy
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s = env.reset()
        done = False
        while not done:
            for t in range(T):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)
                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done)) # prob[a].item() -> a의 확률값. 나중에 ratio 계산할 때 쓰임

                score += r
                if done:
                    break
            model.train()