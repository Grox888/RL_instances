import numpy as np
import torch
from torch import nn


class Critic_Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Critic_Network, self).__init__()

        def _init_weights(x):
            if isinstance(x, nn.Linear):
                x.weight.data.normal_(0, 0.1)

        self.layers = nn.Sequential(*[
            nn.Linear(input_dim, 2 * output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * output_dim, 2 * output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * output_dim, output_dim)
        ])
        self.layers.apply(_init_weights)

    def forward(self, x):
        return self.layers(x)


class DQN:
    def __init__(self, env, Epoch=1000, lr=0.01, epsilon=1, experience_buffer_length=1024, behavior_update=1024,
                 target_update=256, batch_size=256):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = env
        self.Epoch = Epoch
        self.lr = lr
        self.epsilon = epsilon
        self.experience_buffer = []
        self.experience_buffer_length = experience_buffer_length
        self.behavior_update = behavior_update
        self.target_update = target_update
        self.batch_size = batch_size
        self.behavior = Critic_Network(2, 5).to(self.device)
        self.policy_optimization()

    def policy(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        a_star = torch.argmax(self.behavior(state), dim=-1).to(torch.device('cpu'))
        return a_star

    def policy_exploration(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        a_star = torch.argmax(self.behavior(state), dim=-1)
        distribution = torch.ones(5, dtype=torch.float32, device=self.device) * self.epsilon / 5
        distribution[a_star] += 1 - self.epsilon
        actions = torch.multinomial(distribution, num_samples=1, replacement=False).to(torch.device('cpu'))
        return actions[0]

    def exp_append(self, s, a, y_T):
        self.experience_buffer.append((*s, a, y_T))
        if len(self.experience_buffer) > self.experience_buffer_length:
            self.experience_buffer.pop(0)
        return True

    def exp_replay(self):
        distribution = np.ones(len(self.experience_buffer), dtype=float) / len(self.experience_buffer)
        buff_idx = [i for i in range(len(self.experience_buffer))]
        exp_buff_idx = np.random.choice(buff_idx, size=self.batch_size, p=distribution, replace=False)
        batch_s = torch.tensor([(self.experience_buffer[i][0], self.experience_buffer[i][1]) for i in exp_buff_idx], dtype=torch.float32, device=self.device)
        batch_a = torch.tensor([[self.experience_buffer[i][2]] for i in exp_buff_idx], dtype=torch.int64, device=self.device)
        batch_y_T = torch.tensor([self.experience_buffer[i][3] for i in exp_buff_idx], dtype=torch.float32, device=self.device)
        return batch_s, batch_a, batch_y_T

    def policy_optimization(self):
        loss = nn.MSELoss(reduction='mean')
        target = Critic_Network(2, 5).to(self.device)
        optimizer = torch.optim.Adam(lr=self.lr, params=target.parameters())
        state = (np.random.randint(0, self.env.width - 1), np.random.randint(0, self.env.length - 1))
        Epoch = self.Epoch
        count = 0
        while Epoch > 0:
            # behavior
            with torch.no_grad():
                reward, period = self.env.behavior(state, self.policy_exploration, n=1)
                action = period[0][1]
                state_next = torch.tensor(period[-1][0], dtype=torch.float32, device=self.device)
                y_T = reward + self.env.discount * target(state_next).detach().max()
                self.exp_append(state, action, y_T)
                state = state_next.cpu().numpy()
            # target update
            count += 1
            if count % self.target_update == 0:
                optimizer.zero_grad()
                batch_s, batch_a, batch_y_T = self.exp_replay()
                y = target(batch_s).gather(1, batch_a).reshape(-1,)
                l = loss(y, batch_y_T)
                l.backward()
                optimizer.step()
                Epoch -= 1
                print(f'Epoch {self.Epoch - Epoch}, loss: {l}')
            # behavior update
            if count % self.behavior_update == 0:
                self.behavior.load_state_dict(target.state_dict())
        self.behavior.load_state_dict(target.state_dict())
        return True


if __name__ == '__main__':

    A = torch.tensor([(1, 2)], dtype=torch.float32)
    print(A.max())