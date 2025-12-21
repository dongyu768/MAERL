import torch
import torch.nn as nn
import torch.nn.functional as F

# define the actor network
class Actor(nn.Module):
    def __init__(self, config, agent_id):
        super(Actor, self).__init__()
        self.max_action = config['high_action']
        self.fc1 = nn.Linear(config['obs_shape'][agent_id], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.action_out = nn.Linear(64, config['action_shape'][agent_id])
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))
        return actions
# define the critic network
class Critic(nn.Module):
    def __init__(self, config):
        super(Critic, self).__init__()
        self.max_action = config['high_action']
        # network architecture of q1
        self.fc1 = nn.Linear(sum(config['obs_shape'])+sum(config['action_shape']), 64)
        self.fc2 = nn.Linear(64, 64)
        self.q_out1 = nn.Linear(64, 1)
        # network architecture of q2
        self.fc3 = nn.Linear(sum(config['obs_shape'])+sum(config['action_shape']), 64)
        self.fc4 = nn.Linear(64, 64)
        self.q_out2 = nn.Linear(64, 1)
    def forward(self, state, action):
        state = torch.cat(state, dim=1)
        for i in range(len(action)):
            action[i] /= self.max_action
        action = torch.cat(action, dim=1)
        x1 = torch.cat([state, action], dim=1)
        x2 = x1
        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        q_value1 = self.q_out1(x1)

        x2 = F.relu(self.fc3(x2))
        x2 = F.relu(self.fc4(x2))
        q_value2 = self.q_out2(x2)
        return q_value1, q_value2
    
    def Q1(self, state, action):
        state = torch.cat(state, dim=1)
        for i in range(len(action)):
            action[i] /= self.max_action
        action = torch.cat(action, dim=1)
        x1 = torch.cat([state, action], dim=1)
        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        q_value1 = self.q_out1(x1)
        return q_value1
