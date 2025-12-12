import numpy as np
from Env import Env
import random

class REDQLearner:

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

    def random_two_number(self):
        random_number_1 = random.randint(0, self.K-1)
        random_number_2 = random.randint(0, self.K-1)
        while random_number_2 == random_number_1:
            random_number_2 = random.randint(0, self.K-1)
        return random_number_1, random_number_2

    def explore(self, state):
        self.Count_S[state[0]][state[1]] += 1
        epsilon_temp = self.epsilon / np.power(self.Count_S[state[0]][state[1]], 0.5)
        self.index_1, self.index_2 = self.random_two_number()
        if np.random.random() >= epsilon_temp:
            action = np.argmax([min(self.Q[self.index_1][state[0]][state[1]][i], self.Q[self.index_2][state[0]][state[1]][i]) for i in range(4)])
        else:
            action = np.random.choice(4)
        return action

    def learning(self, state, action, reward, next_state, done):
        Y = reward
        if not done:
            Y += self.gamma * max([min(self.Q[self.index_1][next_state[0]][next_state[1]][i], self.Q[self.index_2][next_state[0]][next_state[1]][i]) for i in range(4)])
        index_update = random.randint(0, self.K-1)
        self.Count_S_A[index_update][state[0]][state[1]][action] += 1
        lr = self.learningRate / np.power(self.Count_S_A[index_update][state[0]][state[1]][action], 0.8)
        self.Q[index_update][state[0]][state[1]][action] += lr * (Y - self.Q[index_update][state[0]][state[1]][action])

    def maxQ(self):
        index_1, index_2 = self.random_two_number()
        return max([min(self.Q[index_1][0][0][i],self.Q[index_2][0][0][i]) for i in range(4)])