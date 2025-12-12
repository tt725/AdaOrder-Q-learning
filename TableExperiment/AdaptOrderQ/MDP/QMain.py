import numpy as np
from Logger import Logger
import time
from Env import Env
from QLearner import QLearner


def QLearning():
    max_Q_A_repeat = np.zeros((10000, 500))
    A_left_P_repeat = np.zeros((10000, 500))
    for repeat in range(10000):
        env = Env()
        agent = QLearner()
        max_Q_A, A_left_P = QUpdate(env=env, agent=agent)
        max_Q_A_repeat[int(repeat)] = max_Q_A
        A_left_P_repeat[int(repeat)] = A_left_P
        if repeat % 1000 == 0:
            log.logger.info("************************************")
            log.logger.info("repeat experiment number is {}".format(repeat))
    log.logger.info("************************************")
    log.logger.info("max_Q_A is: \n{}".format(list(max_Q_A_repeat.mean(axis=0))))
    log.logger.info("A_left_P is: \n{}".format(list(A_left_P_repeat.mean(axis=0))))


def QUpdate(env, agent):
    max_Q_A = np.zeros(500)
    A_left_P = np.zeros(500)
    for epoch in range(500):
        A_visit = 0.0
        A_left = 0.0
        state = env.reset()
        while True:
            if state == env.STATE_A:
                A_visit += 1.0
            action = agent.explore(state)
            if state == env.STATE_A and action == env.Left:
                A_left += 1.0
            next_state, reward, done = env.step(action)
            agent.learning(state, action, reward, next_state, done)
            if done:
                break
            state = next_state
        max_Q_A[int(epoch)] = agent.maxQ(env.STATE_A)
        A_left_P[int(epoch)] = A_left / A_visit
    return max_Q_A, A_left_P


if __name__ == "__main__":
    log = Logger(
        './Result/log.' + "Q" + " " + (time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime())),
        level='debug')
    QLearning()

