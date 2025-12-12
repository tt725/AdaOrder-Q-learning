import numpy as np
from Logger import Logger
import time
from Env import Env
from OrderQLearner import OrderQLearner

NUMBER_ESTIMATOR = 16
NUMBER_ORDER = 1

def OrderQLearning():
    max_Q_S_repeat = np.zeros((1000, 100))
    for repeat in range(1000):
        env = Env()
        agent = OrderQLearner(number_estimator=NUMBER_ESTIMATOR, number_order=NUMBER_ORDER)
        max_Q_S= OrderQUpdate(env=env, agent=agent)
        max_Q_S_repeat[int(repeat)] = max_Q_S
        if repeat % 100 == 0:
            log.logger.info("************************************")
            log.logger.info("repeat experiment number is {}".format(repeat))
    log.logger.info("************************************")
    log.logger.info("max_Q_S is: \n{}".format(list(max_Q_S_repeat.mean(axis=0))))


def OrderQUpdate(env, agent):
    max_Q_S = np.zeros(100)
    state = env.reset()
    for step in range(10000):
        if step % 100 == 0:
            max_Q_S[step // 100] = agent.maxQ(env.STATE_S)
        action = agent.explore(state)
        next_state, reward, done = env.step(action)
        agent.learning(state, action, reward, next_state, done)
        state = next_state
    return max_Q_S


if __name__ == "__main__":
    log = Logger(
        './Result/log.' + "OrderQ(" + str(NUMBER_ESTIMATOR) + "," + str(NUMBER_ORDER) + ")"
        + " " + (time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime())),
        level='debug')
    OrderQLearning()

