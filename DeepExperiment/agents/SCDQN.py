from agents.DQN import DQN


class SCDQN(DQN):
  '''
  Implementation of Self-correcting DQN with target network and replay buffer
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    self.SCQ_parameter = cfg['agent']['scq_parameter']

  def compute_q_target(self, next_states, rewards, dones):
    tempQ = self.Q_net[0](next_states).detach() - self.SCQ_parameter * (self.Q_net[0](next_states).detach() - self.Q_net_target[0](next_states).detach())
    q_next = tempQ.max(1)[0]
    q_target = rewards + self.discount * q_next * (1 - dones)
    return q_target