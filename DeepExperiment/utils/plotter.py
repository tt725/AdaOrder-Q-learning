import os
import numpy as np
import pandas as pd
import seaborn as sns; sns.set(style="ticks"); sns.set_context("paper") #sns.set_context("talk")
from utils.sweeper import Sweeper


class Plotter(object):
  def __init__(self, env_name, merged, x_label, y_label, ci, EMA, runs):
    self.exp = env_name
    self.merged = merged
    self.x_label = x_label
    self.y_label = y_label
    self.ci = ci
    self.EMA = EMA
    self.runs = runs
    self.total_combination = get_total_combination(self.exp)

  def merge_index(self, config_idx, mode):
    result_list = []
    for _ in range(self.runs):
      result_file = f'../logs/{self.exp}/{config_idx}/result_{mode}.feather'
      result = read_file(result_file)
      if result is not None:
        result['Config Index'] = config_idx
        result_list.append(result)
      config_idx += self.total_combination
    if self.EMA:
      xs, ys = [], []
      for result in result_list:
        xs.append(result[self.x_label].to_numpy())
        ys.append(result[self.y_label].to_numpy())
      low = max(x[0] for x in xs)
      high = min(x[-1] for x in xs)
      n = min(len(x) for x in xs)
      for i in range(len(xs)):
        new_x, new_y, _ = symmetric_ema(xs[i], ys[i], low, high, n)
        result_list[i] = result_list[i][:n]
        result_list[i].loc[:, self.x_label] = new_x
        result_list[i].loc[:, self.y_label] = new_y
    else:
      n = min(len(result) for result in result_list)
      for i in range(len(result_list)):
        result_list[i] = result_list[i][:n]

    return result_list

  def get_result(self, exp, config_idx, mode):
    if self.merged == True:
      print(f'[{self.exp}]: Merge {mode} results: {config_idx}/{self.total_combination}')
      result_list = self.merge_index(config_idx, mode)
      return result_list
    else:
      result_file = f'./logs/{exp}/{config_idx}/result_{mode}.feather'
      result = read_file(result_file)
      return [result]

  def result_indexList(self, indexList, mode):
    expIndexModeList = []
    for x in indexList:
      expIndexModeList.append([self.exp, x, mode])
    return self.result_expIndexModeList(expIndexModeList)

  def result_expIndexModeList(self, expIndexModeList):
    results = []
    for exp, config_idx, mode in expIndexModeList:
      print(f'[{exp}]: Plot {mode} results: {config_idx}')
      result_list = self.get_result(exp, config_idx, mode)
      results.append(result_list)
    return results


def one_sided_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1.0, low_counts_threshold=0.0):
  low = xolds[0] if low is None else low
  high = xolds[-1] if high is None else high

  assert xolds[0] <= low, 'low = {} < xolds[0] = {} - extrapolation not permitted!'.format(low, xolds[0])
  assert xolds[-1] >= high, 'high = {} > xolds[-1] = {}  - extrapolation not permitted!'.format(high, xolds[-1])
  assert len(xolds) == len(yolds), 'length of xolds ({}) and yolds ({}) do not match!'.format(len(xolds), len(yolds))

  xolds, yolds = xolds.astype('float64'), yolds.astype('float64')
  luoi = 0  # last unused old index
  sum_y = 0.
  count_y = 0.
  xnews = np.linspace(low, high, n)
  decay_period = (high - low) / (n - 1) * decay_steps
  interstep_decay = np.exp(- 1. / decay_steps)
  sum_ys = np.zeros_like(xnews)
  count_ys = np.zeros_like(xnews)
  for i in range(n):
    xnew = xnews[i]
    sum_y *= interstep_decay
    count_y *= interstep_decay
    while True:
      if luoi >= len(xolds): break
      xold = xolds[luoi]
      if xold <= xnew:
        decay = np.exp(- (xnew - xold) / decay_period)
        sum_y += decay * yolds[luoi]
        count_y += decay
        luoi += 1
      else:
        break
    sum_ys[i] = sum_y
    count_ys[i] = count_y

  ys = sum_ys / count_ys
  ys[count_ys < low_counts_threshold] = np.nan
  return xnews, ys, count_ys


def symmetric_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1.0, low_counts_threshold=0.0):
  xs, ys1, count_ys1 = one_sided_ema(xolds, yolds, low, high, n, decay_steps, low_counts_threshold)
  _, ys2, count_ys2 = one_sided_ema(-xolds[::-1], yolds[::-1], -high, -low, n, decay_steps, low_counts_threshold)
  ys2 = ys2[::-1]
  count_ys2 = count_ys2[::-1]
  count_ys = count_ys1 + count_ys2
  ys = (ys1 * count_ys1 + ys2 * count_ys2) / count_ys
  ys[count_ys < low_counts_threshold] = np.nan
  xs = [int(x) for x in xs]
  return xs, ys, count_ys


def get_total_combination(exp):
  '''
  Get total combination of experiment configuration
  '''
  config_file = f'../configs/{exp}.json'
  assert os.path.isfile(config_file), f'[{exp}]: No config file <{config_file}>!'
  sweeper = Sweeper(config_file)
  return sweeper.config_dicts['num_combinations']


def read_file(result_file):
  if not os.path.isfile(result_file):
    print(f'[No such file <{result_file}>')
    return None
  else:
    return pd.read_feather(result_file)