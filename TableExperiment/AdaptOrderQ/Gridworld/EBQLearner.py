import numpy as np
from Env import Env
import random


class EBQLearner:

    def __init__(self, number_estimator, epsilon=1.0, gamma=0.95, learningRate=1.0):
        self.K = number_estimator
        self.learningRate = learningRate
        self.epsilon = epsilon
        self.gamma = gamma
        self.init_Q_table()

    def init_Q_table(self):
        self.Q = [np.random.normal(0, 0.01, size=(Env().world_size, Env().world_size, 4)) for i in range(self.K)]
        self.Count_S_A = np.zeros(shape=(self.K, Env().world_size, Env().world_size, 4))
        self.Count_S = np.zeros(shape=(Env().world_size, Env().world_size))

    def explore(self, state):
        self.Count_S[state[0]][state[1]] += 1
        epsilon_temp = self.epsilon / np.power(self.Count_S[state[0]][state[1]], 0.5)
        if np.random.random() >= epsilon_temp:
            action = np.argmax([(self.Q[0][state[0]][state[1]][i] + self.Q[1][state[0]][state[1]][i] + self.Q[2][state[0]][state[1]][i] + self.Q[3][state[0]][state[1]][i] +
                                 self.Q[4][state[0]][state[1]][i] + self.Q[5][state[0]][state[1]][i] + self.Q[6][state[0]][state[1]][i] + self.Q[7][state[0]][state[1]][i]) / self.K for i in range(4)])
        else:
            action = np.random.choice(4)
        return action

    def learning(self, state, action, reward, next_state, done):
        Y = reward
        index_update = random.randint(0, self.K-1)
        self.Count_S_A[index_update][state[0]][state[1]][action] += 1
        lr = self.learningRate / np.power(self.Count_S_A[index_update][state[0]][state[1]][action], 1.0)
        if not done:
            action_index = np.argmax(self.Q[index_update][next_state[0]][next_state[1]][:])
            add_q = 0
            for i in range(self.K):
                if i != index_update:
                    add_q += self.Q[i][next_state[0]][next_state[1]][action_index]
            Y += self.gamma * add_q / (self.K-1)
        self.Q[index_update][state[0]][state[1]][action] += lr * (Y - self.Q[index_update][state[0]][state[1]][action])

    def maxQ(self):
        return max([(self.Q[0][0][0][i] + self.Q[1][0][0][i] + self.Q[2][0][0][i] + self.Q[3][0][0][i] +
                     self.Q[4][0][0][i] + self.Q[5][0][0][i] + self.Q[6][0][0][i] + self.Q[7][0][0][i]) / self.K for
                    i in range(4)])