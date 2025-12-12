import numpy as np
from Env import Env


class MaxminQLearner:
    def __init__(self, number_estimator, epsilon=1.0, gamma=0.95, learningRate=1.0):
        self.number_estimator = number_estimator
        self.learningRate = learningRate
        self.epsilon = epsilon
        self.gamma = gamma
        self.init_Q_table()

    def init_Q_table(self):
        self.Q = np.random.normal(0, 0.01, size=(self.number_estimator, Env().nState, Env().nAction))
        self.Count_S_A = np.zeros(shape=(self.number_estimator, Env().nState, Env().nAction))
        self.Count_S = np.zeros(shape=Env().nState)

    def explore(self, state):
        self.Count_S[state] += 1
        epsilon_temp = self.epsilon / np.power(self.Count_S[state], 0.5)
        action_number = Env().action_number(state)
        if np.random.random() >= epsilon_temp:
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
                allQ.append(min(tempQ))
            Y += self.gamma * max(allQ)
        index = np.random.randint(0, self.number_estimator)
        self.Count_S_A[index][state][action] += 1
        lr = self.learningRate / np.power(self.Count_S_A[index][state][action], 0.8)
        self.Q[index][state][action] += lr * (Y - self.Q[index][state][action])

    def maxQ(self, state):
        action_number = Env().action_number(state)
        allQ = []
        for i in range(action_number):
            tempQ = []
            for j in range(self.number_estimator):
                tempQ.append(self.Q[j][state][i])
            allQ.append(np.mean(tempQ))
        return max(allQ)