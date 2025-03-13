import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical


class Actor_Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor_Network, self).__init__()

        def _init_weights(x):
            if isinstance(x, nn.Linear):
                x.weight.data.normal_(0, 0.1)

        self.layers = nn.Sequential(*[
            nn.Linear(input_dim, 2 * output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * output_dim, 2 * output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * output_dim, output_dim),
        ])

        self.layers.apply(_init_weights)

    def forward(self, x):
        return F.softmax(self.layers(x), dim=-1)


class REINFORCE:
    def __init__(self, env, lr=0.0001, epsilon=0.1, Epoch=300):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epsilon = epsilon
        self.env = env
        self.lr = lr
        self.Epoch = Epoch
        self.actor = Actor_Network(2, 5).to(self.device)
        self.policy_optimization()

    def policy(self, state):
        s = torch.tensor(state, dtype=torch.float32, device=self.device)
        a = torch.argmax(self.actor(s), dim=-1).to(torch.device('cpu'))
        return a

    # def policy_exploration(self, state):
    #     s = torch.tensor(state, dtype=torch.float32, device=self.device)
    #     distribution = Categorical(self.actor(s))
    #     a = distribution.sample()
    #     log_p = distribution.log_prob(a)
    #     a.to(torch.device('cpu'))
    #     return a, log_p

    def policy_exploration(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        probs = self.actor(state)
        a_star = torch.argmax(probs, dim=-1)
        distribution = torch.ones(5, dtype=torch.float32, device=self.device) * self.epsilon / 5
        distribution[a_star] += 1 - self.epsilon
        actions = torch.multinomial(distribution, num_samples=1, replacement=False)
        action = actions[0]
        log_prob = torch.log(probs[action])
        return actions[0], log_prob

    def policy_optimization(self):
        optimizer = torch.optim.Adam(lr=self.lr, params=self.actor.parameters())
        Epoch = self.Epoch
        for i in range(Epoch):
            optimizer.zero_grad()
            # state = (np.random.randint(0, self.env.width - 1), np.random.randint(0, self.env.length - 1))
            state = (0, 0)
            Return, period = self.env.behavior(state, self.policy_exploration, need_prob=True)
            g = 0
            gama = self.env.discount
            for j in range(len(period) - 2, -1, -1):
                _, _, r, log_p = period[j]
                g = g * gama + r
                loss = -log_p * g
                loss.backward()
            optimizer.step()
            print(f'Epoch {i}, Return: {Return}')





