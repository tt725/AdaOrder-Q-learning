import numpy as np
from Env import Env


class QLearner:
    def __init__(self, epsilon=0.1, gamma=1.0, learningRate=0.2):
        self.learningRate = learningRate
        self.epsilon = epsilon
        self.gamma = gamma
        self.init_Q_table()

    def init_Q_table(self):
        self.Q = np.random.normal(0, 0.01, size=(Env().nState, Env().nAction))

    def explore(self, state):
        action_number = Env().action_number(state)
        if np.random.random() >= self.epsilon:
            action = np.argmax(self.Q[state][:action_number])
        else:
            action = np.random.choice(action_number)
        return action

    def learning(self, state, action, reward, next_state, done):
        Y = reward
        if not done:
            action_number = Env().action_number(next_state)
            Y += self.gamma * max(self.Q[next_state][:action_number])
        self.Q[state][action] += self.learningRate * (Y - self.Q[state][action])

    def Q_Left(self, state):
        return self.Q[state][Env().Left]