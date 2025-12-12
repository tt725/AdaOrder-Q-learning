import numpy as np
from Env import Env
import random


class AdaEQLearner:

    def __init__(self, number_estimator, epsilon=1.0, gamma=0.95, learningRate=1.0):
        self.K = number_estimator
        self.learningRate = learningRate
        self.epsilon = epsilon
        self.gamma = gamma
        self.init_Q_table()
        self.M = number_estimator
        self.c = 0.3

    def init_Q_table(self):
        self.Q = [np.random.normal(0, 0.01, size=(Env().nState, Env().nAction)) for i in range(self.K)]
        self.Count_S_A = np.zeros(shape=(self.K, Env().nState, Env().nAction))
        self.Count_S = np.zeros(shape=Env().nState)

    def number_Q(self, state, action, MC_reward_state_action):
        tempError = []
        for j in range(self.K):
            tempError.append(self.Q[j][state][action] - MC_reward_state_action)
        error = np.std(tempError)
        if (self.M + 1) <= self.K -1 and error > self.c:
            self.M = np.random.randint(self.M + 1, self.K)
            return self.M
        elif (self.M - 1) >= 2 + 1 and error < self.c:
            self.M = np.random.randint(2, self.M-1)
            return self.M
        else:
            return self.M

    def explore(self, state):
        self.Count_S[state] += 1
        epsilon_temp = self.epsilon / np.power(self.Count_S[state], 0.5)
        action_number = Env().action_number(state)
        index_choose = random.sample([i for i in range(self.K)], self.M)
        if np.random.random() >= epsilon_temp:
            allQ = []
            for i in range(action_number):
                tempQ = []
                for j in index_choose:
                    tempQ.append(self.Q[j][state][i])
                allQ.append(min(tempQ))
            action = np.argmax(allQ)
        else:
            action = np.random.choice(action_number)
        return action

    def learning(self, state, action, reward, next_state, done, MC_reward_state_action):
        self.M = self.number_Q(state, action, MC_reward_state_action)
        index_choose = random.sample([i for i in range(self.K)], self.M)
        Y = reward
        if not done:
            action_number = Env().action_number(next_state)
            allQ = []
            for i in range(action_number):
                tempQ = []
                for j in index_choose:
                    tempQ.append(self.Q[j][next_state][i])
                allQ.append(min(tempQ))
            Y += self.gamma * max(allQ)
        index = np.random.randint(0, self.K)
        self.Count_S_A[index][state][action] += 1
        lr = self.learningRate / np.power(self.Count_S_A[index][state][action], 0.8)
        self.Q[index][state][action] += lr * (Y - self.Q[index][state][action])

    def maxQ(self, state):
        index_choose = random.sample([i for i in range(self.K)], self.M)
        action_number = Env().action_number(state)
        allQ = []
        for i in range(action_number):
            tempQ = []
            for j in index_choose:
                tempQ.append(self.Q[j][state][i])
            allQ.append(min(tempQ))
        return max(allQ)
