import numpy as np
from Env import Env
import random


class EBQLearner:

    def __init__(self, number_estimator, epsilon=0.1, gamma=1.0, learningRate=0.1):
        self.K = number_estimator
        self.learningRate = learningRate
        self.epsilon = epsilon
        self.gamma = gamma
        self.init_Q_table()

    def init_Q_table(self):
        self.Q = [np.random.normal(0, 0.01, size=(Env().nState, Env().nAction)) for i in range(self.K)]

    def explore(self, state):
        action_number = Env().action_number(state)
        if np.random.random() >= self.epsilon:
            action = np.argmax([(self.Q[0][state][i] + self.Q[1][state][i] + self.Q[2][state][i] + self.Q[3][state][i] +
                                 self.Q[4][state][i] + self.Q[5][state][i] + self.Q[6][state][i] + self.Q[7][state][i]) / self.K for i in range(action_number)])
        else:
            action = np.random.choice(action_number)
        return action

    def learning(self, state, action, reward, next_state, done):
        Y = reward
        index_update = random.randint(0, self.K -1)
        if not done:
            action_number = Env().action_number(next_state)
            action_index = np.argmax(self.Q[index_update][next_state][:action_number])
            add_q = 0
            for i in range(self.K):
                if i != index_update:
                    add_q += self.Q[i][next_state][action_index]
            Y += self.gamma * add_q / (self.K-1)
        self.Q[index_update][state][action] += self.learningRate * (Y - self.Q[index_update][state][action])

    def maxQ(self, state):
        action_number = Env().action_number(state)
        return max([(self.Q[0][state][i] + self.Q[1][state][i] + self.Q[2][state][i] + self.Q[3][state][i] +
                     self.Q[4][state][i] + self.Q[5][state][i] + self.Q[6][state][i] + self.Q[7][state][i]) / self.K for
                    i in range(action_number)])