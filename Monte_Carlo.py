import numpy as np


class Monte_Carlo:
    def __init__(self, env, threshold=1e-3, epsilon=0.1):
        self.env = env
        self.threshold = threshold
        self.epsilon = epsilon
        self.policy_parameters = np.ones((self.env.width, self.env.length, len(self.env.action_space)), dtype=float) / len(self.env.action_space)
        self.q_value = np.zeros((self.env.width, self.env.length, len(self.env.action_space)), dtype=float)
        self.vist_num = np.zeros((self.env.width, self.env.length, len(self.env.action_space)), dtype=int)
        self.policy_optimization()

    def policy_exploration(self, s):
        samples = np.random.choice([i for i in range(len(self.env.action_space))], size=1, p=self.policy_parameters[s])
        if len(samples) == 0:
            a = 4
        else:
            a = samples[0]
        return a

    def policy(self, s):
        a = np.argmax(self.policy_parameters[s])
        return a

    def policy_optimization(self):
        error = np.inf
        while error > self.threshold:
            state_start = (np.random.randint(0, self.env.width - 1), np.random.randint(0, self.env.length - 1))
            # policy evaluation
            _, episode = self.env.behavior(state_start, self.policy_exploration)
            q_value_before = self.q_value.copy()
            G = 0
            for i in range(len(episode) - 2, -1, -1):
                state, action, r = episode[i]
                G = r + G * self.env.discount
                self.vist_num[(*state, action)] += 1
                k = self.vist_num[(*state, action)]
                q = self.q_value[(*state, action)]
                self.q_value[(*state, action)] = ((k - 1) * q + G) / k
            # policy update
            for i in range(self.policy_parameters.shape[0]):
                for j in range(self.policy_parameters.shape[1]):
                    a_star = np.argmax(self.q_value[i, j])
                    self.policy_parameters[i, j, :] = self.epsilon / self.policy_parameters.shape[-1]
                    self.policy_parameters[i, j, a_star] += 1 - self.epsilon
            error = np.linalg.norm((q_value_before - self.q_value).reshape(-1, 5), ord=1)
            print(error)
        return True
