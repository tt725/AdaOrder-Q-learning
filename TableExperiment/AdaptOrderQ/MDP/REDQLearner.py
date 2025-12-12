import numpy as np
from Env import Env
import random

class REQLearner:

    def __init__(self, number_estimator, epsilon=0.1, gamma=1.0, learningRate=0.1):
        self.K = number_estimator
        self.learningRate = learningRate
        self.epsilon = epsilon
        self.gamma = gamma
        self.init_Q_table()

    def init_Q_table(self):
        self.Q = [np.random.normal(0, 0.01, size=(Env().nState, Env().nAction)) for i in range(self.K)]

    def random_two_number(self):
        random_number_1 = random.randint(0, self.K-1)
        random_number_2 = random.randint(0, self.K-1)
        while random_number_2 == random_number_1:
            random_number_2 = random.randint(0, self.K-1)
        return random_number_1, random_number_2

    def explore(self, state):
        action_number = Env().action_number(state)
        self.index_1, self.index_2 = self.random_two_number()
        if np.random.random() >= self.epsilon:
            action = np.argmax([min(self.Q[self.index_1][state][i], self.Q[self.index_2][state][i]) for i in range(action_number)])
        else:
            action = np.random.choice(action_number)
        return action

    def learning(self, state, action, reward, next_state, done):
        Y = reward
        if not done:
            action_number = Env().action_number(next_state)
            Y += self.gamma * max([min(self.Q[self.index_1][next_state][i], self.Q[self.index_2][next_state][i]) for i in range(action_number)])
        index_update = random.randint(0, self.K-1)
        self.Q[index_update][state][action] += self.learningRate * (Y - self.Q[index_update][state][action])

    def maxQ(self, state):
        action_number = Env().action_number(state)
        index_1, index_2 = self.random_two_number()
        return max([min(self.Q[index_1][state][i],self.Q[index_2][state][i]) for i in range(action_number)])