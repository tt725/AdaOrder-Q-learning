from utils.helper import *
from agents.MaxminDQN import MaxminDQN


class AdaOrderDQN(MaxminDQN):
    '''
  Implementation of AdaOrder DQN with target network and replay buffer

  In the original paper, all Q_nets are updated in Order DQN for every update.
  However, this makes training really slow. Instead, we randomly choose one to update.
  '''

    def __init__(self, cfg):
        super().__init__(cfg)
        self.target_number = self.k

    def compute_q_target(self, next_states, rewards, dones):
        if self.target_number == 1:
            q_next = self.Q_net_target[np.random.choice(list(range(self.k)))](next_states).detach().max(1)[0]
        else:
            q_list = torch.zeros(size=[self.k, self.batch_size, self.get_action_size()]).to(self.device)
            for i in range(0, self.k):
                q_list[i] = self.Q_net_target[i](next_states).detach()
            q_mean = torch.mean(q_list, dim=0)

            q_order_list = torch.zeros(size=[self.target_number, self.batch_size, self.get_action_size()]).to(
                self.device)
            q_order_index = np.random.choice(list(range(self.k)), self.target_number, replace=False)
            for i in range(0, self.target_number):
                q_order_list[i] = q_list[q_order_index[i]]
            q_order_list, _ = torch.sort(q_order_list, dim=0)

            q_adaorder = torch.where(q_order_list < q_mean, q_order_list, q_order_list[0]).max(
                dim=0).values
            q_next = q_adaorder.max(1)[0]

        q_target = rewards + self.discount * q_next * (1 - dones)
        return q_target

    def get_action_selection_q_values(self, state):
        self.target_number = int(self.k * (1-self.step_count / (self.train_steps + 1)))
        if self.target_number == 1:
            q_adaorder = self.Q_net[np.random.choice(list(range(self.k)))](state)
        else:
            q_list = torch.zeros(size=[self.k, 1, self.get_action_size()]).to(self.device)
            for i in range(0, self.k):
                q_list[i] = self.Q_net[i](state)
            q_mean = torch.mean(q_list, dim=0)

            q_order_list = torch.zeros(size=[self.target_number, 1, self.get_action_size()]).to(
                self.device)
            q_order_index = np.random.choice(list(range(self.k)), self.target_number, replace=False)
            for i in range(0, self.target_number):
                q_order_list[i] = q_list[q_order_index[i]]
            q_order_list, _ = torch.sort(q_order_list, dim=0)

            q_adaorder = torch.where(q_order_list < q_mean, q_order_list, q_order_list[0]).max(
                dim=0).values

        q_adaorder = to_numpy(q_adaorder).flatten()
        return q_adaorder
