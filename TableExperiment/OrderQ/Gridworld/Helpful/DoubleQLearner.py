import numpy as np
from Env import Env


class DoubleQLearner:
    def __init__(self, epsilon=1.0, gamma=0.95, learningRate=0.1):
        self.epsilon = epsilon
        self.gamma = gamma
        self.learningRate = learningRate
        self.init_Q_table()

    def init_Q_table(self):
        self.Q1 = np.random.normal(0, 0.01, size=(Env().world_size, Env().world_size, 4))
        self.Q2 = np.random.normal(0, 0.01, size=(Env().world_size, Env().world_size, 4))
        self.Count_S = np.zeros((Env().world_size, Env().world_size))

    def explore(self, state):
        self.Count_S[state[0]][state[1]] += 1
        epsilon_temp = self.epsilon / np.power(self.Count_S[state[0]][state[1]], 0.5)
        if np.random.random() >= epsilon_temp:
            Q3 = [(self.Q1[state[0]][state[1]][i] + self.Q2[state[0]][state[1]][i]) / 2.0 for i in range(4)]
            action = np.argmax(Q3[:])
        else:
            action = np.random.choice(4)
        return action

    def learning(self, state, action, reward, next_state, done):
        Y = reward
        if np.random.random() >= 0.5:
            if not done:
                Y += self.gamma * self.Q2[next_state[0]][next_state[1]][np.argmax(self.Q1[next_state[0]][next_state[1]][:])]
            self.Q1[state[0]][state[1]][action] += self.learningRate * (Y - self.Q1[state[0]][state[1]][action])
        else:
            if not done:
                Y += self.gamma * self.Q1[next_state[0]][next_state[1]][np.argmax(self.Q2[next_state[0]][next_state[1]][:])]
            self.Q2[state[0]][state[1]][action] += self.learningRate * (Y - self.Q2[state[0]][state[1]][action])

    def maxQ(self):
        return max([(self.Q1[0][0][i] + self.Q2[0][0][i]) / 2.0 for i in range(4)])