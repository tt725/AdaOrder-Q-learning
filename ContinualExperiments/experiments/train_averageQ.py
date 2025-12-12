import gym
import numpy as np
import torch
import time
import sys
from diffalgos.algos.average import AverageQAgent
from diffalgos.algos.core import mbpo_epoches, test_agent
from diffalgos.utils.run_utils import setup_logger_kwargs
from diffalgos.utils.logx import EpochLogger

def avergaeQ(env_name, seed=0, epochs='mbpo', steps_per_epoch=1000,
             max_ep_len=1000, n_evals_per_epoch=1,
             logger_kwargs=dict(),
             # following are agent related hyperparameters
             hidden_sizes=(256, 256), replay_size=int(1e6), batch_size=256,
             lr=3e-4, gamma=0.99, polyak=0.995,
             alpha=0.2, auto_alpha=True, target_entropy='mbpo',
             start_steps=5000, delay_update_steps='auto',
             utd_ratio=20, num_Q=10,
             policy_update_delay=20, reseed_each_epoch=True
             ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if epochs == 'mbpo' or epochs < 0:
        epochs = mbpo_epoches[env_name]
    total_steps = steps_per_epoch * epochs + 1

    """set up logger"""
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    """set up environment and seeding"""
    env_fn = lambda: gym.make(env_name)
    env, test_env = env_fn(), env_fn()

    torch.manual_seed(seed)
    np.random.seed(seed)

    def seed_all(epoch):
        seed_shift = epoch * 9999
        mod_value = 999999
        env_seed = (seed + seed_shift) % mod_value
        test_env_seed = (seed + 10000 + seed_shift) % mod_value
        torch.manual_seed(env_seed)
        np.random.seed(env_seed)
        env.seed(env_seed)
        env.action_space.np_random.seed(env_seed)
        test_env.seed(test_env_seed)
        test_env.action_space.np_random.seed(test_env_seed)
    seed_all(epoch=0)

    """prepare to init agent"""
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_ep_len = env._max_episode_steps if max_ep_len > env._max_episode_steps else max_ep_len
    act_limit = env.action_space.high[0].item()
    start_time = time.time()
    sys.stdout.flush()

    """init agent and start training"""
    agent = AverageQAgent(env_name, obs_dim, act_dim, act_limit, device,
                 hidden_sizes, replay_size, batch_size,
                 lr, gamma, polyak,
                 alpha, auto_alpha, target_entropy,
                 start_steps, delay_update_steps,
                 utd_ratio, num_Q,
                 policy_update_delay)

    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    for t in range(total_steps):
        a = agent.get_exploration_action(o, env)
        o2, r, d, _ = env.step(a)
        ep_len += 1
        d = False if ep_len == max_ep_len else d
        agent.store_data(o, a, r, o2, d)
        agent.train(logger)
        o = o2
        ep_ret += r

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        if (t+1) % steps_per_epoch == 0:
            epoch = t // steps_per_epoch
            test_agent(agent, test_env, max_ep_len, logger) # add logging here
            if reseed_each_epoch:
                seed_all(epoch)

            """logging"""
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()

            sys.stdout.flush()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Ant-v2')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=-1) # -1 means use mbpo epochs
    parser.add_argument('--algorithm_name', type=str, default='average')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--num_Q', type=int, default=10)
    parser.add_argument('--utd_ratio', type=int, default=1)
    parser.add_argument('--policy_update_delay', type=int, default=20)
    args = parser.parse_args()

    # modify the code here if you want to use a different naming scheme
    exp_name_full = '%s' % args.env + '_%s' % args.algorithm_name +\
                    '_N%d' % args.num_Q

    logger_kwargs = setup_logger_kwargs(exp_name_full, args.seed, args.data_dir)

    avergaeQ(args.env, seed=args.seed, epochs=args.epochs,
             logger_kwargs=logger_kwargs,
         num_Q=args.num_Q,
         utd_ratio=args.utd_ratio,
         policy_update_delay=args.policy_update_delay)
