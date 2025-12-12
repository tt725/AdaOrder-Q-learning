import numpy as np
import math
from Env import Env


class SoftmaxQLearner:

    def __init__(self, epsilon=0.1, gamma=1.0, learningRate=0.1):
        self.learningRate = learningRate
        self.epsilon = epsilon
        self.gamma = gamma
        self.init_Q_table()

    def init_Q_table(self):
        self.Q = np.random.normal(0, 0.01, size=(Env().nState, Env().nAction))

    def explore(self, state):
        action_number = Env().action_number(state)
        if np.random.random() >= self.epsilon:
            action = np.argmax(self.Q[state][:action_number])
        else:
            action = np.random.choice(action_number)
        return action

    def learning(self, state, action, reward, next_state, done):
        Y = reward
        if not done:
            action_number = Env().action_number(next_state)
            z_exp = [math.exp(2.0 * i) for i in self.Q[next_state][:action_number]]
            sum_z_exp = sum(z_exp)
            W = [i / sum_z_exp for i in z_exp]
            func = lambda x, y: x * y
            result = map(func, self.Q[next_state][:action_number], W)
            list_result = list(result)
            Y += self.gamma * sum(list_result)
        self.Q[state][action] += self.learningRate * (Y - self.Q[state][action])

    def maxQ(self, state):
        action_number = Env().action_number(state)
        return max(self.Q[state][:action_number])