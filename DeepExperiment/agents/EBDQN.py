from utils.helper import *
from agents.MaxminDQN import MaxminDQN


class EBDQN(MaxminDQN):
  '''
  Implementation of Ensemble Bootstrapping DQN with target network and replay buffer
  
  In the original paper, all Q_nets are updated in Ensemble Bootstrapping DQN for every update.
  However, this makes training really slow. Instead, we randomly choose one to update.
  '''
  def __init__(self, cfg):
    super().__init__(cfg)

  def learn(self):
    super().learn()
    if (self.step_count // self.sgd_update_frequency) % self.target_network_update_freqency == 0:
      for i in range(self.k):
        self.Q_net_target[i].load_state_dict(self.Q_net[i].state_dict())

  def compute_q_target(self, next_states, rewards, dones):
    self.update_Q_net_index = np.random.choice(list(range(self.k)))
    best_actions = self.Q_net_target[self.update_Q_net_index](next_states).detach().argmax(1).unsqueeze(1)
    EB_index_list = list(range(self.k))
    EB_index_list.remove(self.update_Q_net_index)
    q_EB = self.Q_net_target[EB_index_list[0]](next_states).detach().gather(1, best_actions).squeeze()
    for i in range(1, self.k-1):
      q = self.Q_net_target[EB_index_list[i]](next_states).detach().gather(1, best_actions).squeeze()
      q_EB = q_EB + q
    q_next = q_EB / (self.k-1)
    q_target = rewards + self.discount * q_next * (1 - dones)
    return q_target
  
  def get_action_selection_q_values(self, state):
    q_EB = self.Q_net[0](state)
    for i in range(1, self.k):
      q = self.Q_net[i](state)
      q_EB = q_EB + q
    q_EB = to_numpy(q_EB / self.k).flatten()
    return q_EB