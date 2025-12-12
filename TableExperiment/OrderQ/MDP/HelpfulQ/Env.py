import numpy as np

class Env():
    def __init__(self):
        self.STATE_A = 0
        self.STATE_B = 1
        self.STATE_T = 2
        self.nA = 2
        self.Left = 0
        self.Right = 1
        self.nB = 10
        self.nT = 1
        self.nState = 3
        self.nAction = 10
        self.state = self.STATE_A

    def reset(self):
        self.state = self.STATE_A
        return self.state

    def step(self, action):
        # A--left
        if self.state == self.STATE_A and action == self.Left:
            self.state = self.STATE_B
            return self.state, 0, False
        # A--right
        elif self.state == self.STATE_A and action == self.Right:
            self.state = self.STATE_T
            return self.state, 0, True
        # B--all actions
        elif self.state == self.STATE_B and action == 0:
            self.state = self.STATE_T
            reward = np.random.normal(0.1, 0.1)
            return self.state, reward, True
        else:
            self.state = self.STATE_T
            reward = np.random.normal(-0.1, 0.1)
            return self.state, reward, True

    def action_number(self, state):
        if state == self.STATE_A:
            return self.nA
        elif state == self.STATE_B:
            return self.nB
        else:
            return self.nT