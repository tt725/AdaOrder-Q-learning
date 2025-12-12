import numpy as np
from Logger import Logger
import time
from Env import Env
from DoubleQLearner import DoubleQLearner


def DoubleQLearning():
    Q_Left_repeat = np.zeros((5000, 500))
    A_right_P_repeat = np.zeros((5000, 500))
    for repeat in range(5000):
        env = Env()
        agent = DoubleQLearner()
        Q_Left, A_right_P = DoubleQUpdate(env=env, agent=agent)
        Q_Left_repeat[int(repeat)] = Q_Left
        A_right_P_repeat[int(repeat)] = A_right_P
        if repeat % 500 == 0:
            log.logger.info("************************************")
            log.logger.info("repeat experiment number is {}".format(repeat))
    log.logger.info("************************************")
    log.logger.info("Q_Left is: \n{}".format(list(Q_Left_repeat.mean(axis=0))))
    log.logger.info("A_right_P is: \n{}".format(list(A_right_P_repeat.mean(axis=0))))


def DoubleQUpdate(env, agent):
    Q_Left = np.zeros(500)
    A_right_P = np.zeros(500)
    for epoch in range(500):
        Q_Left[int(epoch)] = agent.Q_Left(env.STATE_A)
        A_visit = 0.0
        A_right = 0.0
        state = env.reset()
        while True:
            if state == env.STATE_A:
                A_visit += 1.0
            action = agent.explore(state)
            if state == env.STATE_A and action == env.Right:
                A_right += 1.0
            next_state, reward, done = env.step(action)
            agent.learning(state, action, reward, next_state, done)
            if done:
                break
            state = next_state
        A_right_P[int(epoch)] = A_right / A_visit
    return Q_Left, A_right_P


if __name__ == "__main__":
    log = Logger(
        './Result/log.' + "DoubleQ" + " " + (time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime())),
        level='debug')
    DoubleQLearning()

