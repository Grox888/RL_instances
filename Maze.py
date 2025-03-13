import numpy as np
import random


class Maze:
    def __init__(self, width, length, blocks, discount=0.6, episode_length=1000):
        self.width = width
        self.length = length
        self.discount = discount
        self.episode_length = episode_length
        self.action_space = {'up': (0, -1),
                             'down': (0, 1),
                             'left': (-1, 0),
                             'right': (1, 0),
                             'stay': (0, 0)}
        self.grid = np.zeros((width, length), dtype=float)
        self.block_index = []
        remain_blocks = blocks
        while remain_blocks > 0:
            h = int(random.random() * width)
            w = int(random.random() * length)
            if self.grid[h, w] == 0:
                self.grid[h, w] = np.inf
                self.block_index.append((h, w))
                remain_blocks -= 1
        self.terminal_state = (-1, -1)
        while self.terminal_state == (-1, -1):
            h = int(random.random() * width)
            w = int(random.random() * length)
            if (h, w) not in self.block_index:
                self.terminal_state = (h, w)
                self.grid[self.terminal_state] = -np.inf

    def forward(self, s, a):
        h, w = self._add(s, self.action_space[a])
        if (h < 0 or h >= self.width) \
                or (w < 0 or w >= self.length) \
                or (h, w) in self.block_index:
            r = -10
        elif (h, w) == self.terminal_state:
            r = 10
        else:
            r = 0
        s_next = (np.clip(h, 0, self.width - 1), np.clip(w, 0, self.length - 1))
        return r, s_next

    def behavior(self, start_position, policy, n=0, need_prob=False):
        actions = ['up', 'down', 'left', 'right', 'stay']
        state = start_position
        if n < 1 or n > self.episode_length:
            hop = self.episode_length
        else:
            hop = n
        period = []
        reward_sum = 0
        gama = 1
        if need_prob:
            while hop > 0:
                action, log_prob = policy(state)
                a = actions[action]
                r, s_next = self.forward(state, a)
                period.append((state, action, r, log_prob))
                reward_sum += gama * r
                gama *= self.discount
                state = s_next
                hop -= 1
            period.append((state, None, None, None))
        else:
            while hop > 0:
                action = policy(state)
                a = actions[action]
                r, s_next = self.forward(state, a)
                period.append((state, action, r))
                reward_sum += gama * r
                gama *= self.discount
                state = s_next
                hop -= 1
            period.append((state, None, None))
        return reward_sum, period

    def show_maze(self):
        print(self.grid)

    def show_opt(self, start_position, policy):
        trajectory = self.grid.copy()
        Return, episode = self.behavior(start_position, policy)
        for coordinate, _, _ in episode:
            trajectory[coordinate] = 1
        print(f'Return = {Return}\nPath:\n{trajectory}')
        return True

    def _add(self, a, b):
        x1, x2 = a
        y1, y2 = b
        return x1 + y1, x2 + y2
