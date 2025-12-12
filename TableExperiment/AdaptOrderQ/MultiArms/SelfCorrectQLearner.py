import numpy as np
from Env import Env


class SelfCorrectQLearner:
    def __init__(self, epsilon=1.0, gamma=0.95, learningRate=1.0, parameter=2.0):
        self.learningRate = learningRate
        self.epsilon = epsilon
        self.gamma = gamma
        self.parameter = parameter
        self.init_Q_table()

    def init_Q_table(self):
        self.Q = np.random.normal(0, 0.01, size=(Env().nState, Env().nAction))
        self.Q_previous = np.random.normal(0, 0.01, size=(Env().nState, Env().nAction))
        self.Count_S_A = np.zeros(shape=(Env().nState, Env().nAction))
        self.Count_S = np.zeros(shape=Env().nState)

    def explore(self, state):
        self.Count_S[state] += 1
        epsilon_temp = self.epsilon / np.power(self.Count_S[state], 0.5)
        action_number = Env().action_number(state)
        if np.random.random() >= epsilon_temp:
            action = np.argmax(self.Q[state][:action_number])
        else:
            action = np.random.choice(action_number)
        return action

    def learning(self, state, action, reward, next_state, done):
        Y = reward
        if not done:
            Q_W = [self.Q[next_state][i] - self.parameter * (self.Q[next_state][i] - self.Q_previous[next_state][i]) for i in
                   range(Env().action_number(next_state))]
            Y += self.gamma * self.Q[next_state][np.argmax(Q_W)]
        self.Q_previous[state][action] = self.Q[state][action]
        self.Count_S_A[state][action] += 1
        lr = self.learningRate / np.power(self.Count_S_A[state][action], 0.8)
        self.Q[state][action] += lr * (Y - self.Q[state][action])

    def maxQ(self, state):
        action_number = Env().action_number(state)
        return max(self.Q[state][:action_number])
