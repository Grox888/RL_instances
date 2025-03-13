import numpy as np


class Sarsa:
    def __init__(self, env, threshold=5e-3, lr=0.001, epsilon=0.05, n=1):
        self.env = env
        self.lr = lr
        self.threshold = threshold
        self.epsilon = epsilon
        self.episode_length = env.episode_length
        self.n = n
        self.policy_parameters = np.ones((self.env.width, self.env.length, len(self.env.action_space)),
                                         dtype=float) / len(self.env.action_space)
        self.q_value = np.zeros((self.env.width, self.env.length, len(self.env.action_space)), dtype=float)
        self.policy_optimization()

    def policy(self, s):
        a = np.argmax(self.policy_parameters[s])
        return a

    def policy_exploration(self, s):
        samples = np.random.choice([i for i in range(len(self.env.action_space))], size=1, p=self.policy_parameters[s])
        return samples[0]

    # def policy_optimization(self):
    #     error = np.inf
    #     while error > self.threshold:
    #         state = (np.random.randint(0, self.env.width - 1), np.random.randint(0, self.env.length - 1))
    #         hop = self.episode_length
    #         q_value_before = self.q_value.copy()
    #         # policy_before = self.policy_parameters.copy()
    #         while (hop - self.n) >= 0:
    #             # policy evaluation
    #             reward_sum, period = self.env.behavior(state, self.policy_exploration, n=self.n)
    #             hop -= self.n
    #             state_next = period[-1][0]
    #             action_next = self.policy_exploration(state_next)
    #             q_hat = reward_sum + self.q_value[(*state_next, action_next)] * (self.env.discount ** self.n)
    #             self.q_value[(*state, period[0][1])] = self.q_value[(*state, period[0][1])] - self.lr * (self.q_value[(*state, period[0][1])] - q_hat)
    #             # policy update
    #             a_star = np.argmax(self.q_value[state])
    #             self.policy_parameters[state[0], state[1], :] = self.epsilon / 5
    #             self.policy_parameters[state[0], state[1], a_star] += 1 - self.epsilon
    #             state = state_next
    #         error = np.linalg.norm((q_value_before - self.q_value).reshape(-1, 5), ord=2)
    #         # error = np.linalg.norm((policy_before - self.policy_parameters).reshape(-1, 5), ord=1)
    #         print(error)
    #     return True

    def policy_optimization(self):
        error = np.inf
        # state = (np.random.randint(0, self.env.width - 1), np.random.randint(0, self.env.length - 1))
        state = (0, 0)
        count = 0
        C = 128
        q_value_before = self.q_value.copy()
        while error > self.threshold:
            reward_sum, period = self.env.behavior(state, self.policy_exploration, n=2)
            action = period[0][1]
            reward = period[0][2]
            state_next = period[1][0]
            action_next = period[1][1]
            q_hat = reward + self.env.discount * self.q_value[(*state_next, action_next)]
            q = self.q_value[(*state, action)].copy()
            self.q_value[(*state, action)] = q + self.lr * (q_hat - q)
            a_star = np.argmax(self.q_value[state[0], state[1], :])
            self.policy_parameters[state[0], state[1], :] = self.epsilon / 5
            self.policy_parameters[state[0], state[1], a_star] += 1 - self.epsilon
            state = period[-1][0]
            count += 1
            if count % C == 0:
                error = np.linalg.norm((q_value_before - self.q_value).reshape(-1, 5), ord=1)
                q_value_before = self.q_value.copy()
                print(error)
        return True






