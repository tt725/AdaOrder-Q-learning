from utils.helper import *
from agents.MaxminDQN import MaxminDQN


class REDQN(MaxminDQN):
  '''
  Implementation of Random Ensemble DQN with target network and replay buffer
  
  In the original paper, all Q_nets are updated in Random Ensemble DQN for every update.
  However, this makes training really slow. Instead, we randomly choose one to update.
  '''
  def __init__(self, cfg):
    super().__init__(cfg)

  def compute_q_target(self, next_states, rewards, dones):
    random_index_list = np.random.choice(list(range(self.k)), size=2, replace=False)
    q_min = torch.min(self.Q_net_target[random_index_list[0]](next_states).clone().detach(), self.Q_net_target[random_index_list[1]](next_states).clone().detach())
    q_next = q_min.max(1)[0]
    q_target = rewards + self.discount * q_next * (1 - dones)
    return q_target
  
  def get_action_selection_q_values(self, state):
    random_index_list = np.random.choice(list(range(self.k)), size=2, replace=False)
    q_min = torch.min(self.Q_net[random_index_list[0]](state), self.Q_net[random_index_list[1]](state))
    q_min = to_numpy(q_min).flatten()
    return q_min