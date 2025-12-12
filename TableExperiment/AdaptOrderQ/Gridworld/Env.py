import numpy as np

class Env():
    def __init__(self, world_size=3, a=-12, b=10, c=-40, d=50):
        self.world_size = world_size
        self.state = [0, 0]

        self.Left = 0
        self.Down = 1
        self.Right = 2
        self.Up = 3

        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def reset(self):
        self.state = [0, 0]
        return self.state

    def state_test(self, state):
        self.state = state
        return self.state

    def step(self, action):
        if self.state[0] == self.world_size - 1 and self.state[1] == self.world_size - 1:
            if np.random.random() >= 0.5:
                reward = self.c
            else:
                reward = self.d
            return self.state, reward, True
        # left
        if action == self.Left:
            act = [-1, 0]
        # down
        if action == self.Down:
            act = [0, -1]
        # right
        elif action == self.Right:
            act = [1, 0]
        # up
        elif action == self.Up:
            act = [0, 1]
        next_state = [self.state[0] + act[0], self.state[1] + act[1]]
        x, y = next_state
        if x < 0 or x >= self.world_size or y < 0 or y >= self.world_size:
            next_state = self.state
        self.state = next_state
        if np.random.random() >= 0.5:
            reward = self.a
        else:
            reward = self.b
        return self.state, reward, False
