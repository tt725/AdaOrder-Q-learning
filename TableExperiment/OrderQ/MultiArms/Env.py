import numpy as np


class Env():
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std
        self.STATE_S = 0
        self.nState = 1
        self.nAction = 10
        self.state = self.STATE_S

    def reset(self):
        self.state = self.STATE_S
        return self.state

    def step(self, action):
        reward = np.random.normal(self.mean, self.std)
        self.state = self.STATE_S
        return self.state, reward, False

    def action_number(self, state):
        return self.nAction