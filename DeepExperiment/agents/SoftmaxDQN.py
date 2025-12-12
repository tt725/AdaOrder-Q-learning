from utils.helper import *
from agents.DQN import DQN


class SoftmaxDQN(DQN):
    '''
  Implementation of Softmax DQN with target network and replay buffer
  '''

    def __init__(self, cfg):
        super().__init__(cfg)
        self.SoftQ_parameter = cfg['agent']['SoftQ_parameter']

    def compute_q_target(self, next_states, rewards, dones):
        q_next = self.Q_net_target[0](next_states).detach()
        q_next_weight = torch.softmax(self.SoftQ_parameter * q_next, dim=-1)
        soft_q = torch.sum(q_next * q_next_weight, dim=1)
        soft_q = torch.where(torch.isnan(soft_q), q_next.max(1)[0], soft_q)
        q_target = rewards + self.discount * soft_q * (1 - dones)
        return q_target