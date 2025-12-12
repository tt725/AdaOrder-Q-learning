from utils.helper import *
from agents.MaxminDQN import MaxminDQN


class AdaAverageDQN(MaxminDQN):
    '''
  Implementation of AdaAverage DQN with target network and replay buffer

  In the original paper, all Q_nets are updated in AdaAverage DQN for every update.
  However, this makes training really slow. Instead, we randomly choose one to update.
  '''

    def __init__(self, cfg):
        super().__init__(cfg)
        self.target_number = self.k

    def compute_q_target(self, next_states, rewards, dones):
        if self.target_number == 1:
            q_next = self.Q_net_target[np.random.choice(list(range(self.k)))](next_states).detach().max(1)[0]
        else:
            mean_index = np.random.choice(list(range(self.k)), self.target_number, replace=False)
            q_ensemble = self.Q_net_target[mean_index[0]](next_states).clone().detach()
            for i in range(1, self.target_number-1):
                q = self.Q_net_target[mean_index[i]](next_states).detach()
                q_ensemble = q_ensemble + q
            q_next = q_ensemble.max(1)[0] / self.target_number

        q_target = rewards + self.discount * q_next * (1 - dones)
        return q_target

    def get_action_selection_q_values(self, state):
        self.target_number = int(self.k * (1 - self.step_count / (self.train_steps + 1)))
        if self.target_number == 1:
            q_adamean = self.Q_net[np.random.choice(list(range(self.k)))](state)
        else:
            mean_index = np.random.choice(list(range(self.k)), self.target_number, replace=False)
            q_adamean = self.Q_net[mean_index[0]](state)
            for i in range(1, self.target_number-1):
                q = self.Q_net[mean_index[i]](state)
                q_adamean = q_adamean + q
            q_adamean = q_adamean / self.target_number
        q_adamean = to_numpy(q_adamean).flatten()
        return q_adamean
