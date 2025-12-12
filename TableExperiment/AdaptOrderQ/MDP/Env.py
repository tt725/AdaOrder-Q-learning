import numpy as np

class Env():
    def __init__(self, mean1=-0.1, mean2=0.1, std1=1, std2=1):
        self.STATE_A = 0
        self.STATE_B = 1
        self.STATE_C = 2
        self.STATE_T = 3
        self.nA = 2
        self.Left = 0
        self.Right = 1
        self.nB = 10
        self.nC = 10
        self.nT = 1
        self.nState = 4
        self.nAction = 10
        self.state = self.STATE_A
        self.mean1 = mean1
        self.mean2 = mean2
        self.std1 = std1
        self.std2 = std2


    def reset(self):
        self.state = self.STATE_A
        return self.state

    def step(self, action):
        # A--left
        if self.state == self.STATE_A and action == self.Left:
            self.state = self.STATE_B
            return self.state, 0, False
        # A--right
        if self.state == self.STATE_A and action == self.Right:
            self.state = self.STATE_C
            return self.state, 0, False
        # B--all actions
        elif self.state == self.STATE_B:
            self.state = self.STATE_T
            reward = np.random.normal(self.mean1, self.std1)
            return self.state, reward, True
        # C--all actions
        elif self.state == self.STATE_C:
            self.state = self.STATE_T
            reward = np.random.normal(self.mean2, self.std2)
            return self.state, reward, True

    def state_test(self, state):
        self.state = state
        return self.state

    def action_number(self, state):
        if state == self.STATE_A:
            return self.nA
        elif state == self.STATE_B:
            return self.nB
        elif state == self.STATE_C:
            return self.nC
        else:
            return self.nT