from algo_models.algo.maddpg import MADDPG
import numpy as np
import torch

class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)
        # self.policy = MADDPGEnsemble(args, agent_id)
    def select_action(self, o, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            u = np.random.uniform(-self.args['high_action'], self.args['high_action'], self.args['action_shape'][self.agent_id])
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            pi = self.policy.actor_network(inputs).squeeze(0)
            # print('{}:{}'.format(self.name, pi))
            u = pi.cpu().numpy()
            noise = noise_rate * self.args['high_action'] * np.random.randn(*u.shape)  # gaussian noise
            u += noise
            u = np.clip(u, -self.args['high_action'], self.args['high_action'])
        return u.copy()
    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)

    # Ensemble select action
    # def select_action(self, o, noise_rate, epsilon, k): # 修改
    #     if np.random.uniform() < epsilon:
    #         u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
    #     else:
    #         inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
    #         pi = self.policy.actor_network[k](inputs).squeeze(0)
    #         # print('{}:{}'.format(self.name, pi))
    #         u = pi.cpu().numpy()
    #         noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
    #         u += noise
    #         u = np.clip(u, -self.args.high_action, self.args.high_action)
    #     return u.copy()
    # def learn(self, transitions, other_agents, k):
    #     self.policy.train(transitions, other_agents, k)



