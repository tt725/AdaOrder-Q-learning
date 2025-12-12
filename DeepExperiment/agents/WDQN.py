from agents.DQN import DQN


class WDQN(DQN):
  '''
  Implementation of Weighted Double DQN with target network and replay buffer
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    self.weight_parameter = cfg['agent']['weight_parameter']

  def compute_q_target(self, next_states, rewards, dones):
    best_actions = self.Q_net[0](next_states).detach().argmax(1).unsqueeze(1)
    q_next = self.weight_parameter * self.Q_net_target[0](next_states).detach().gather(1, best_actions).squeeze() \
             + (1-self.weight_parameter) * self.Q_net_target[0](next_states).detach().max(1)[0]
    q_target = rewards + self.discount * q_next * (1 - dones)
    return q_target