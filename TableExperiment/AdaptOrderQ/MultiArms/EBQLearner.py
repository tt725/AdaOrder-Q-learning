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
        self.Q = [np.random.normal(0, 0.01, size=(Env().nState, Env().nAction)) for i in range(self.K)]
        self.Count_S_A = np.zeros(shape=(self.K, Env().nState, Env().nAction))
        self.Count_S = np.zeros(shape=Env().nState)

    def explore(self, state):
        self.Count_S[state] += 1
        epsilon_temp = self.epsilon / np.power(self.Count_S[state], 0.5)
        action_number = Env().action_number(state)
        if np.random.random() >= epsilon_temp:
            action = np.argmax([(self.Q[0][state][i] + self.Q[1][state][i] + self.Q[2][state][i] + self.Q[3][state][i] +
                                 self.Q[4][state][i] + self.Q[5][state][i] + self.Q[6][state][i] + self.Q[7][state][i]) / self.K for i in range(action_number)])
        else:
            action = np.random.choice(action_number)
        return action

    def learning(self, state, action, reward, next_state, done):
        Y = reward
        index_update = random.randint(0, self.K-1)
        self.Count_S_A[index_update][state][action] += 1
        lr = self.learningRate / np.power(self.Count_S_A[index_update][state][action], 0.8)
        if not done:
            action_number = Env().action_number(next_state)
            action_index = np.argmax(self.Q[index_update][next_state][:action_number])
            add_q = 0
            for i in range(self.K):
                if i != index_update:
                    add_q += self.Q[i][next_state][action_index]
            Y += self.gamma * add_q / (self.K-1)
        self.Q[index_update][state][action] += lr * (Y - self.Q[index_update][state][action])

    def maxQ(self, state):
        action_number = Env().action_number(state)
        return max([(self.Q[0][state][i] + self.Q[1][state][i] + self.Q[2][state][i]+ self.Q[3][state][i] +
                     self.Q[4][state][i] + self.Q[5][state][i] + self.Q[6][state][i] + self.Q[7][state][i]) / self.K for i in range(action_number)])
