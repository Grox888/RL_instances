import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical


class Critic_Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Critic_Network, self).__init__()

        def _init_weights(x):
            if isinstance(x, nn.Linear):
                x.weight.data.normal_(0, 0.1)

        self.layers = nn.Sequential(*[
            nn.Linear(input_dim, 4 * input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * input_dim, 2 * input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * input_dim, output_dim)
        ])
        self.layers.apply(_init_weights)

    def forward(self, x):
        return self.layers(x)


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
            nn.Softmax(dim=-1)
        ])
        self.layers.apply(_init_weights)

    def forward(self, x):
        return self.layers(x)


class A2C_off_policy:
    def __init__(self, env, lr=1e-3, Epoch=1200, epsilon=1, exp_buff_length=2048, behavior_update=256, target_update=256, batch_size=256):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = env
        self.lr = lr
        self.Epoch = Epoch
        self.epsilon = epsilon
        self.exp_buff_length = exp_buff_length
        self.exp_buff = []
        self.behavior_update = behavior_update
        self.target_update = target_update
        self.batch_size = batch_size
        self.actor = Actor_Network(2, 5).to(self.device)
        self.critic = Critic_Network(2, 1).to(self.device)
        self.policy_optimization()

    def policy(self, state):
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device)
            a = torch.argmax(self.actor(s), dim=-1).to(torch.device('cpu'))
        return a

    def policy_exploration(self, state):
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device)
            probs = self.actor(s).detach()
            a_star = torch.argmax(probs, dim=-1)
            distribution = torch.ones(5, dtype=torch.float32, device=self.device) * self.epsilon / 5
            distribution[a_star] += 1 - self.epsilon
            actions = torch.multinomial(distribution, num_samples=1, replacement=False)
            a = actions[0]
            p = distribution[a]
        return a.to(torch.device('cpu')), p.to(torch.device('cpu'))

    def exp_append(self, s, a, r, s_, p):
        self.exp_buff.append((*s, a, r, *s_, p))
        if len(self.exp_buff) > self.exp_buff_length:
            self.exp_buff.pop(0)
        return True

    def exp_replay(self):
        adv_ratio = 0.5
        """
        增加优质样本被选中的可能性
        """
        distribution = torch.softmax(torch.tensor([elem[3] for elem in self.exp_buff], dtype=torch.float32, device=self.device), dim=-1)
        distribution = distribution * adv_ratio + (1 - adv_ratio) / self.exp_buff_length
        exp_buff_idx = torch.multinomial(distribution, num_samples=self.batch_size, replacement=True)
        # distribution = np.ones(len(self.exp_buff), dtype=float) / len(self.exp_buff)
        # buff_idx = [i for i in range(len(self.exp_buff))]
        # exp_buff_idx = np.random.choice(buff_idx, size=self.batch_size, p=distribution, replace=False)
        batch_s = torch.tensor([(self.exp_buff[i][0], self.exp_buff[i][1]) for i in exp_buff_idx], dtype=torch.float32, device=self.device)
        batch_a = torch.tensor([[self.exp_buff[i][2]] for i in exp_buff_idx], dtype=torch.int64, device=self.device)
        batch_r = torch.tensor([self.exp_buff[i][3] for i in exp_buff_idx], dtype=torch.float32, device=self.device)
        batch_s_ = torch.tensor([(self.exp_buff[i][4], self.exp_buff[i][5]) for i in exp_buff_idx], dtype=torch.float32, device=self.device)
        batch_p = torch.tensor([self.exp_buff[i][6] for i in exp_buff_idx], dtype=torch.float32, device=self.device)
        return batch_s, batch_a, batch_r, batch_s_, batch_p

    def critic_loss(self, batch_s_value, batch_y_T, batch_adv_weight):
        l_vector = 0.5 * batch_adv_weight * (batch_s_value - batch_y_T) ** 2
        return torch.sum(l_vector) / self.batch_size

    def actor_loss(self, batch_td_error, batch_adv_weight, batch_p):
        l_vector = -batch_adv_weight * batch_td_error * torch.log(batch_p)
        return torch.sum(l_vector) / self.batch_size

    def policy_optimization(self):
        target_actor = Actor_Network(2, 5).to(self.device)
        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        actor_optimizer = torch.optim.Adam(target_actor.parameters(), lr=self.lr)
        Epoch = self.Epoch
        state = (np.random.randint(0, self.env.width - 1), np.random.randint(0, self.env.length - 1))
        count = 0
        while Epoch > 0:
            # behavior
            with torch.no_grad():
                reward, period = self.env.behavior(state, self.policy_exploration, n=1, need_prob=True)
                _, a, r, p = period[0]
                s_ = period[-1][0]
                self.exp_append(state, a, r, s_, p)
                state = s_
            count += 1
            # Target update
            if count % self.target_update == 0:
                batch_s, batch_a, batch_r, batch_s_, batch_p0 = self.exp_replay()
                batch_p1 = target_actor(batch_s).detach().gather(1, batch_a).reshape(-1,)
                batch_adv_weight = batch_p1 / batch_p0
                # critic update
                critic_optimizer.zero_grad()
                batch_s_value = self.critic(batch_s).reshape(-1,)
                batch_y_T = batch_r + self.env.discount * self.critic(batch_s_).detach().reshape(-1,)
                critic_l = self.critic_loss(batch_s_value, batch_y_T, batch_adv_weight)
                critic_l.backward()
                critic_optimizer.step()
                # actor_update
                batch_td_error = batch_y_T - batch_s_value.detach()
                actor_optimizer.zero_grad()
                batch_p = target_actor(batch_s).gather(1, batch_a).reshape(-1,)
                actor_l = self.actor_loss(batch_td_error, batch_adv_weight, batch_p)
                actor_l.backward()
                actor_optimizer.step()
                Epoch -= 1
                # validation
                with torch.no_grad():
                    Return, _ = self.env.behavior((0, 0), self.policy)
                    print(f'Epoch{self.Epoch - Epoch}, Return: {Return:.5f}, Critic_Loss: {critic_l:.5f}, Actor_Loss: {actor_l:.5f}, MIN_p1: {batch_p1.min()}')
            # Behavior update
            if count % self.behavior_update == 0:
                self.actor.load_state_dict(target_actor.state_dict())
        self.actor.load_state_dict(target_actor.state_dict())
        return True



