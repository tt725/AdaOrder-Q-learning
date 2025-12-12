import numpy as np
from Env import Env
import random


def orderQ(Q_all, M):
    Q = random.sample(Q_all, M)
    return np.mean(Q)


class AdaAverageQLearner:
    def __init__(self, number_estimator, parameter, epsilon=0.1, gamma=1.0, learningRate=0.1):
        self.parameter = parameter
        self.number_estimator = number_estimator
        self.learningRate = learningRate
        self.epsilon = epsilon
        self.gamma = gamma
        self.init_Q_table()

    def init_Q_table(self):
        self.Q = np.random.normal(0, 0.01, size=(self.number_estimator, Env().nState, Env().nAction))
        self.Count_S_A = np.zeros(shape=(self.number_estimator, Env().nState, Env().nAction))

    def explore(self, state):
        action_number = Env().action_number(state)
        if np.random.random() >= self.epsilon:
            allQ = []
            for i in range(action_number):
                tempQ = []
                for j in range(self.number_estimator):
                    tempQ.append(self.Q[j][state][i])
                allQ.append(np.mean(tempQ))
            action = np.argmax(allQ)
        else:
            action = np.random.choice(action_number)
        return action

    def learning(self, state, action, reward, next_state, done):
        index = np.random.randint(0, self.number_estimator)
        self.Count_S_A[index][state][action] += 1
        Y = reward
        if not done:
            action_number = Env().action_number(next_state)
            allQ = []
            for i in range(action_number):
                temp = int(
                    self.number_estimator * 1.0 / np.power(self.Count_S_A[index][next_state][i] + 1, self.parameter))
                if temp < 1:
                    temp = 1
                tempQ = []
                for j in range(self.number_estimator):
                    tempQ.append(self.Q[j][next_state][i])
                allQ.append(orderQ(tempQ, temp))
            Y += self.gamma * max(allQ)
        self.Q[index][state][action] += self.learningRate * (Y - self.Q[index][state][action])

    def maxQ(self, state):
        action_number = Env().action_number(state)
        allQ = []
        for i in range(action_number):
            tempQ = []
            for j in range(self.number_estimator):
                tempQ.append(self.Q[j][state][i])
            allQ.append(np.mean(tempQ))
        return max(allQ)
