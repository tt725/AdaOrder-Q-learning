import json

class Sweeper(object):
  def __init__(self, config_file):
    with open(config_file, 'r') as f:
      self.config_dicts = json.load(f)
    self.get_num_combinations_of_dict(self.config_dicts)

  def get_num_combinations_of_dict(self, config_dict):
    assert type(config_dict) == dict, 'Config file must be a dict!'
    num_combinations_of_dict = 1
    for key, values in config_dict.items():
      num_combinations_of_list = self.get_num_combinations_of_list(values)
      num_combinations_of_dict *= num_combinations_of_list
    config_dict['num_combinations'] = num_combinations_of_dict

  def get_num_combinations_of_list(self, config_list):
    assert type(config_list) == list, 'Elements in a config dict must be a list!'
    num_combinations_of_list = 0
    for value in config_list:
      if type(value) == dict:
        if not('num_combinations' in value.keys()):
          self.get_num_combinations_of_dict(value)
        num_combinations_of_list += value['num_combinations']
      else:
        num_combinations_of_list += 1
    return num_combinations_of_list

  def generate_config_for_idx(self, idx):
    cfg = self.get_dict_value(self.config_dicts, (idx-1) % self.config_dicts['num_combinations'])
    cfg['config_idx'] = idx
    cfg['num_combinations'] = self.config_dicts['num_combinations']
    return cfg

  def get_list_value(self, config_list, idx):
    for value in config_list:
      if type(value) == dict:
        if idx + 1 - value['num_combinations'] <= 0:
          return self.get_dict_value(value, idx)
        else:
          idx -= value['num_combinations']
      else:
        if idx == 0:
          return value
        else:
          idx -= 1
  
  def get_dict_value(self, config_dict, idx):
    cfg = dict()
    for key, values in config_dict.items():
      if key == 'num_combinations':
        continue
      num_combinations_of_list = self.get_num_combinations_of_list(values)
      value = self.get_list_value(values, idx % num_combinations_of_list)
      cfg[key] = value
      idx = idx // num_combinations_of_list
    return cfg
  
  def print_config_dict(self, config_dict):
    cfg_json = json.dumps(config_dict, indent=2)
    print(cfg_json, end='\n')
