'''Defined Various Network Architecture'''
"""World Model actor."""
import torch
import torch.nn as nn
import torch.nn.functional as F

# define the actor network
class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.max_action = args['high_action']
        self.fc1 = nn.Linear(args['obs_shape'][agent_id], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, args['action_shape'][agent_id])
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))
        return actions
# define the critic network
class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.max_action = args['high_action']
        self.fc1 = nn.Linear(sum(args['obs_shape'][:args['n_agents']])+sum(args['action_shape'][:args['n_agents']]), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)
    def forward(self, state, action):
        state = torch.cat(state, dim=1)
        for i in range(len(action)):
            action[i] /= self.max_action
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value
# 定义近似网络
class Approx(nn.Module):
    def __init__(self, args, agent_id):
        super(Approx, self).__init__()
        self.max_action = args['high_action']
        self.fc1 = nn.Linear(args['obs_shape'][agent_id], 64)
        self.fc2 = nn.Linear(64, 64)
        # self.fc3 = nn.Linear(64, 64)
        self.action_out = nn.Linear(64, args['action_shape'][agent_id])
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)  # 确保 x 是 Tensor 类型
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        action_probs = F.softmax(self.action_out(x), dim=-1)  # 使用 softmax 输出概率分布
        return action_probs 
    def sample(self, state):
        action_probs = self.forward(state)
        action = torch.multinomial(action_probs, 1)  # 从概率分布中采样一个动作
        return action
    def log_prob(self, state, action):
        action_probs = self.forward(state)
        # 通过选择动作的索引来获得该动作的概率值
        # 将 action 转换为 LongTensor 类型
        # action = action.long()
        # print('action_probs \n',action_probs.shape)
        # print('action \n',action.shape)
        # action_prob = action_probs.gather(1, action)  # 选择对应动作的概率
        # print(action_prob.shape)
        log_prob = torch.log(action_probs + 1e-10)  # 为避免log(0)加一个小的常数
        return log_prob
    def entropy(self, state):
        action_probs = self.forward(state)
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=-1)  # 计算熵
        return entropy



