import numpy as np


class Iterative_Method:

    def __init__(self, env, error=1e-6, truncate=3):
        self.env = env
        self.error = 1e-6
        self.truncate = 3
        self.policy_parameters = self.policy_optimization()

    def policy(self, s):
        return self.policy_parameters[s]

    def policy_optimization(self):
        error = self.error
        truncate = self.truncate
        puzzle = self.env
        discount = puzzle.discount
        state_value = np.zeros((puzzle.width, puzzle.length), dtype=float)
        action_value = np.zeros((puzzle.width, puzzle.length, len(puzzle.action_space)), dtype=float)
        policy = np.zeros((puzzle.width, puzzle.length), dtype=int)
        actions = ['up', 'down', 'left', 'right', 'stay']
        flag_convergence = 0
        while flag_convergence == 0:
            # Policy evaluation
            for k in range(truncate):
                for i in range(puzzle.width):
                    for j in range(puzzle.length):
                        s = (i, j)
                        a = actions[policy[s]]
                        r, s_next = puzzle.forward(s, a)
                        temp = r + discount * state_value[s_next]
                        if abs(temp - state_value[s]) <= error:
                            flag_convergence = 1
                        else:
                            flag_convergence = 0
                        state_value[s] = temp
            # action_value update & policy update
            for i in range(puzzle.width):
                for j in range(puzzle.length):
                    for k in range(len(puzzle.action_space)):
                        s = (i, j)
                        a = actions[k]
                        r, s_next = puzzle.forward(s, a)
                        action_value[i, j, k] = r + discount * state_value[s_next]
                    policy[i, j] = np.argmax(action_value[i, j])
        return policy
