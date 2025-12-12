import numpy as np
from Env import Env


class DoubleQLearner:
    def __init__(self, epsilon=0.1, gamma=1.0, learningRate=0.1):
        self.learningRate = learningRate
        self.epsilon = epsilon
        self.gamma = gamma
        self.init_Q_table()

    def init_Q_table(self):
        self.Q1 = np.random.normal(0, 0.01, size=(Env().nState, Env().nAction))
        self.Q2 = np.random.normal(0, 0.01, size=(Env().nState, Env().nAction))

    def explore(self, state):
        action_number = Env().action_number(state)
        if np.random.random() >= self.epsilon:
            Q3 = [(self.Q1[state][i] + self.Q2[state][i]) / 2.0 for i in range(action_number)]
            action = np.argmax(Q3[:])
        else:
            action = np.random.choice(action_number)
        return action

    def learning(self, state, action, reward, next_state, done):
        Y = reward
        if np.random.random() >= 0.5:
            if not done:
                action_number = Env().action_number(next_state)
                Y += self.gamma * self.Q2[next_state][np.argmax(self.Q1[next_state][:action_number])]
            self.Q1[state][action] += self.learningRate * (Y - self.Q1[state][action])
        else:
            if not done:
                action_number = Env().action_number(next_state)
                Y += self.gamma * self.Q1[next_state][np.argmax(self.Q2[next_state][:action_number])]
            self.Q2[state][action] += self.learningRate * (Y - self.Q2[state][action])

    def Q_Left(self, state):
        return (self.Q1[state][Env().Left] + self.Q2[state][Env().Left]) / 2.0
