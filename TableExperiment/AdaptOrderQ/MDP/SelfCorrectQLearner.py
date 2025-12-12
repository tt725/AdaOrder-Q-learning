import numpy as np
from Env import Env


class SelfCorrectQLearner:
    def __init__(self, epsilon=0.1, gamma=1.0, learningRate=0.1, parameter=2.0):
        self.learningRate = learningRate
        self.epsilon = epsilon
        self.gamma = gamma
        self.parameter = parameter
        self.init_Q_table()

    def init_Q_table(self):
        self.Q = np.random.normal(0, 0.01, size=(Env().nState, Env().nAction))
        self.Q_previous = np.random.normal(0, 0.01, size=(Env().nState, Env().nAction))

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
            Q_W = [self.Q[next_state][i] - self.parameter * (self.Q[next_state][i] - self.Q_previous[next_state][i]) for i in
                   range(Env().action_number(next_state))]
            Y += self.gamma * self.Q[next_state][np.argmax(Q_W)]
        self.Q_previous[state][action] = self.Q[state][action]
        self.Q[state][action] += self.learningRate * (Y - self.Q[state][action])

    def maxQ(self, state):
        action_number = Env().action_number(state)
        return max(self.Q[state][:action_number])
