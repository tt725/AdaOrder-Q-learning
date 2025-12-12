import numpy as np
from Env import Env


def orderQ(Q, m):
    b = sorted(Q)
    return b[m-1]


class OrderQLearner:
    def __init__(self, number_estimator, number_order, epsilon=0.1, gamma=1.0, learningRate=0.1):
        self.number_order = number_order
        self.number_estimator = number_estimator
        self.learningRate = learningRate
        self.epsilon = epsilon
        self.gamma = gamma
        self.init_Q_table()

    def init_Q_table(self):
        self.Q = np.random.normal(0, 0.01, size=(self.number_estimator, Env().nState, Env().nAction))

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
        Y = reward
        if not done:
            action_number = Env().action_number(next_state)
            allQ = []
            for i in range(action_number):
                tempQ = []
                for j in range(self.number_estimator):
                    tempQ.append(self.Q[j][next_state][i])
                allQ.append(orderQ(tempQ, self.number_order))
            Y += self.gamma * max(allQ)
        index = np.random.randint(0, self.number_estimator)
        self.Q[index][state][action] += self.learningRate * (Y - self.Q[index][state][action])

    def Q_Left(self, state):
        tempQ = []
        for j in range(self.number_estimator):
            tempQ.append(self.Q[j][state][Env().Left])
        return np.mean(tempQ)
