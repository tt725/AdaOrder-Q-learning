import numpy as np
from Logger import Logger
import time
from Env import Env
from AdaOrderQLearner import AdaOrderQLearner

NUMBER_ESTIMATOR = 4
PARAMETER = 0.1

def OrderQLearning():
    max_Q_A_repeat = np.zeros((10000, 500))
    A_left_P_repeat = np.zeros((10000, 500))
    for repeat in range(10000):
        env = Env()
        agent = AdaOrderQLearner(number_estimator=NUMBER_ESTIMATOR, parameter=PARAMETER)
        max_Q_A, A_left_P = OrderQUpdate(env=env, agent=agent)
        max_Q_A_repeat[int(repeat)] = max_Q_A
        A_left_P_repeat[int(repeat)] = A_left_P
        if repeat % 1000 == 0:
            log.logger.info("************************************")
            log.logger.info("repeat experiment number is {}".format(repeat))
    log.logger.info("************************************")
    log.logger.info("max_Q_A is: \n{}".format(list(max_Q_A_repeat.mean(axis=0))))
    log.logger.info("A_left_P is: \n{}".format(list(A_left_P_repeat.mean(axis=0))))

def OrderQUpdate(env, agent):
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
        './Result/log.' + "AdaOrderQ(" + str(NUMBER_ESTIMATOR) + "," + str(PARAMETER) + ")"
        + " " + (time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime())),
        level='debug')
    OrderQLearning()

