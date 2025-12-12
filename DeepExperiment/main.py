import os
import argparse

from utils.sweeper import Sweeper
from utils.helper import make_dir
from experiment import Experiment


def args_parse():
  parser = argparse.ArgumentParser(description="Deep reinforcement learning by tantao:")
  parser.add_argument('--config_file', type=str, default='./configs/minatar_Asterix.json',
                      help='Configuration file for the chosen model')
  parser.add_argument('--config_idx', type=int, default=1,
                      help='Configuration index')
  parser.add_argument('--cuda', type=str, default='1',
                      help="choose the index of cuda")
  args = parser.parse_args()
  return args


def main():
  args = args_parse()
  os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

  sweeper = Sweeper(args.config_file)
  cfg = sweeper.generate_config_for_idx(args.config_idx)
  cfg.setdefault('history_length', 4)
  cfg.setdefault('epsilon_decay', 0.999)
  cfg.setdefault('sgd_update_frequency', 1)
  cfg['env'].setdefault('max_episode_steps', -1)
  cfg.setdefault('show_tb', False)
  cfg.setdefault('render', False)
  cfg.setdefault('gradient_clip', -1)
  cfg['exp'] = args.config_file.split('/')[-1].split('.')[0]
  logs_dir = f"./logs/{cfg['exp']}/{cfg['config_idx']}/"
  train_log_path = logs_dir + 'result_Train.feather'
  test_log_path = logs_dir + 'result_Test.feather'
  model_path = logs_dir + 'model.pt'
  cfg_path = logs_dir + 'config.json'
  cfg['logs_dir'] = logs_dir
  cfg['train_log_path'] = train_log_path
  cfg['test_log_path'] = test_log_path
  cfg['model_path'] = model_path
  cfg['cfg_path'] = cfg_path
  make_dir(cfg['logs_dir'])

  exp = Experiment(cfg)
  exp.run()

if __name__=='__main__':
  main()