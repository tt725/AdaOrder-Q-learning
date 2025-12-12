from utils.helper import *
from agents.MaxminDQN import MaxminDQN


class AdaEDQN(MaxminDQN):
  '''
  Implementation of Adaptive Ensemble DQN with target network and replay buffer
  
  In the original paper, all Q_nets are updated in Adaptive Ensemble DQN for every update.
  However, this makes training really slow. Instead, we randomly choose one to update.
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    self.M = cfg['agent']['target_networks_num']
    self.c = 0.3

  def compute_q_target(self, next_states, rewards, dones):
    tempError = torch.zeros(size=[self.k]).to(self.device)
    for j in range(self.k):
      tempError[j] = torch.max(self.Q_net_target[j](to_tensor(self.state, device=self.device).unsqueeze(0))) - to_tensor(self.total_episode_reward, device=self.device)
    error = torch.std(tempError)
    if (self.M + 1) <= self.k - 1 and to_tensor(error, device=self.device) > to_tensor(self.c, device=self.device):
      self.M = np.random.randint(self.M + 1, self.k)
    elif (self.M - 1) >= 2 + 1 and to_tensor(error, device=self.device) < to_tensor(self.c, device=self.device):
      self.M = np.random.randint(2, self.M - 1)
    random_index_list = np.random.choice(list(range(self.k)), size=self.M, replace=False)
    q_min = self.Q_net_target[random_index_list[0]](next_states).clone().detach()
    for i in range(1, self.M):
      q = self.Q_net_target[random_index_list[i]](next_states).detach()
      q_min = torch.min(q_min, q)
    q_next = q_min.max(1)[0]
    q_target = rewards + self.discount * q_next * (1 - dones)
    return q_target
  
  def get_action_selection_q_values(self, state):
    random_index_list = np.random.choice(list(range(self.k)), size=self.M, replace=False)
    q_min = self.Q_net[random_index_list[0]](state)
    for i in range(1, self.M):
      q = self.Q_net[random_index_list[i]](state)
      q_min = torch.min(q_min, q)
    q_min = to_numpy(q_min).flatten()
    return q_min