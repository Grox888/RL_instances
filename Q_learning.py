import numpy as np


class Q_learning:
    def __init__(self, env, threshold=1e-6, lr=0.1, epsilon=0.5, experience_buffer_length=1024, behavior_update=1024, target_update=64, batch_size=64):
        self.env = env
        self.threshold = threshold
        self.lr = lr
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.behavior_update = behavior_update
        self.target_update = target_update
        self.experience_buffer = []
        self.experience_buffer_length = experience_buffer_length
        self.q_value = np.zeros((self.env.width, self.env.length, 5), dtype=float)
        self.policy_optimization()

    def policy(self, state):
        return np.argmax(self.q_value[state])

    def policy_exploration(self, state):
        a_star = np.argmax(self.q_value[state])
        action_distribution = np.ones(5, dtype=float) * self.epsilon / 5
        action_distribution[a_star] += 1 - self.epsilon
        samples = np.random.choice([i for i in range(5)], size=1, p=action_distribution)
        return samples[0]

    def exp_append(self, s, a, r, s_):
        self.experience_buffer.append((s, a, r, s_))
        if len(self.experience_buffer) > self.experience_buffer_length:
            self.experience_buffer.pop(0)
        return True

    def exp_replay(self):
        distribution = np.ones(len(self.experience_buffer), dtype=float) / len(self.experience_buffer)
        exp_buff_idx = [i for i in range(len(self.experience_buffer))]
        return np.random.choice(exp_buff_idx, size=self.batch_size, p=distribution, replace=False)

    def policy_optimization(self):
        error = np.inf
        state = (np.random.randint(0, self.env.width - 1), np.random.randint(0, self.env.length - 1))
        q_value_target = self.q_value.copy()
        q_value_before = self.q_value.copy()
        count = 0
        while error > self.threshold and error > 0:
            # behavior
            reward, period = self.env.behavior(state, self.policy_exploration, n=1)
            action = period[0][1]
            state_next = period[-1][0]
            self.exp_append(state, action, reward, state_next)
            state = state_next
            # target update
            count += 1
            if count % self.target_update == 0:
                samples = self.exp_replay()
                for idx in samples:
                    s, a, r, s_ = self.experience_buffer[idx]
                    q = q_value_target[(*s, a)]
                    q_hat = r + self.env.discount * np.max(q_value_target[s_])
                    q_value_target[(*s, a)] = q + self.lr * (q_hat - q)
                # error evaluation
                error = np.linalg.norm((q_value_target - q_value_before).reshape(-1, 5), ord=2)
                print(f'{count / self.target_update} update, error: {error}')
                q_value_before = q_value_target.copy()
            # behavior update
            if count % self.behavior_update == 0:
                self.q_value = q_value_target.copy()
        self.q_value = q_value_target
        return True





