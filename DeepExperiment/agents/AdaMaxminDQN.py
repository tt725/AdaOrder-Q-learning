from utils.helper import *
from agents.MaxminDQN import MaxminDQN


class AdaMaxminDQN(MaxminDQN):
    '''
  Implementation of AdaMaxmin DQN with target network and replay buffer

  In the original paper, all Q_nets are updated in AdaMaxmin DQN for every update.
  However, this makes training really slow. Instead, we randomly choose one to update.
  '''

    def __init__(self, cfg):
        super().__init__(cfg)
        self.target_number = self.k

    def compute_q_target(self, next_states, rewards, dones):
        if self.target_number == 1:
            q_next = self.Q_net_target[np.random.choice(list(range(self.k)))](next_states).detach().max(1)[0]
        else:
            min_index = np.random.choice(list(range(self.k)), self.target_number, replace=False)
            q_min = self.Q_net_target[min_index[0]](next_states).clone().detach()
            for i in range(1, self.target_number-1):
                q = self.Q_net_target[min_index[i]](next_states).detach()
                q_min = torch.min(q_min, q)
            q_next = q_min.max(1)[0]
        q_target = rewards + self.discount * q_next * (1 - dones)
        return q_target

    def get_action_selection_q_values(self, state):
        self.target_number = int(self.k * (1 - self.step_count / (self.train_steps + 1)))
        if self.target_number == 1:
            q_adamin = self.Q_net[np.random.choice(list(range(self.k)))](state)
        else:
            min_index = np.random.choice(list(range(self.k)), self.target_number, replace=False)
            q_adamin = self.Q_net[min_index[0]](state)
            for i in range(1, self.target_number-1):
                q = self.Q_net[min_index[i]](state)
                q_adamin = torch.min(q_adamin, q)
        q_adamin = to_numpy(q_adamin).flatten()
        return q_adamin
