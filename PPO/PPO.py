class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        self.gamma = 0.98
        self.lmbda = 0.95
        self.eps = 0.1
        self.K = 2

        self.fc1 = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_v = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr = 0.0005)

    def pi(self, x, softmax_dim = 0): # softmax_dim은 batch 처리를 위해 필요함. batch 단위 학습할 땐 dim = 1
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim = softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_list, s_prime_lst, prob_a_list, done_list = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s,a,r,s_prime,done_mask,prob_a = torch.tensor(s_lst, dtype = torch.float), torch.tensor(a_lst), torch.tensor(r_lst), torch.tensor(s_prime_lstm, dtype = torch.float), torch.tensor(done_lst, dtype = torch.float), torch.tensor(prob_a_lst)
        self. data = []
        return s,a,r,s_prime,done_mask,prob_a

    def train_net(self):
        s,a,r,s_prime,done_mask,prob_a = self.make_batch()

