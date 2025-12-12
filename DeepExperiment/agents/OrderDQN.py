from utils.helper import *
from agents.MaxminDQN import MaxminDQN


class OrderDQN(MaxminDQN):
    '''
  Implementation of Order DQN with target network and replay buffer

  In the original paper, all Q_nets are updated in Order DQN for every update.
  However, this makes training really slow. Instead, we randomly choose one to update.
  '''

    def __init__(self, cfg):
        super().__init__(cfg)
        self.order_num = cfg['agent']['order_num']  # number of target networks

    def compute_q_target(self, next_states, rewards, dones):
        q_order_list = torch.zeros(size=[self.k, self.batch_size, self.get_action_size()]).to(self.device)
        for i in range(0, self.k):
            q = self.Q_net_target[i](next_states).detach()
            q_order_list[i] = q
        q_order_list, _ = torch.sort(q_order_list, dim=0)
        q_next = q_order_list[self.order_num - 1].max(1)[0]
        q_target = rewards + self.discount * q_next * (1 - dones)
        return q_target

    def get_action_selection_q_values(self, state):
        q_order_list = torch.zeros(size=[self.k, 1, self.get_action_size()]).to(self.device)
        for i in range(0, self.k):
            q = self.Q_net[i](state)
            q_order_list[i] = q
        q_order_list, _ = torch.sort(q_order_list, dim=0)
        q_order = to_numpy(q_order_list[self.order_num - 1]).flatten()
        return q_order
