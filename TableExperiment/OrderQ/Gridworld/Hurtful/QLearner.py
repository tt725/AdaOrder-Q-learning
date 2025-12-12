import numpy as np
from Env import Env


class QLearner:
    def __init__(self, epsilon=1.0, gamma=0.95, learningRate=1.0):
        self.epsilon = epsilon
        self.gamma = gamma
        self.learningRate = learningRate
        self.init_Q_table()

    def init_Q_table(self):
        self.Q = np.random.normal(0, 0.01, size=(Env().world_size, Env().world_size, 4))
        self.Count_S_A = np.zeros((Env().world_size, Env().world_size, 4))
        self.Count_S = np.zeros((Env().world_size, Env().world_size))

    def explore(self, state):
        self.Count_S[state[0]][state[1]] += 1
        epsilon_temp = self.epsilon / np.power(self.Count_S[state[0]][state[1]], 0.5)
        if np.random.random() >= epsilon_temp:
            action = np.argmax(self.Q[state[0]][state[1]][:])
        else:
            action = np.random.choice(4)
        return action

    def learning(self, state, action, reward, next_state, done):
        self.Count_S_A[state[0]][state[1]][action] += 1
        lr = self.learningRate / np.power(self.Count_S_A[state[0]][state[1]][action], 0.8)
        Y = reward
        if not done:
            Y += self.gamma * max(self.Q[next_state[0]][next_state[1]][:])
        self.Q[state[0]][state[1]][action] += lr * (Y - self.Q[state[0]][state[1]][action])

    def maxQ(self):
        return max(self.Q[0][0][:])
