'''Defined Various Agents Classes'''
from algorithms.maddpg.maddpg import MADDPG
from core.agents.BaseAgent import BaseAgent
import numpy as np
import torch
import os

class MADDPGAgent(BaseAgent):
    """
    MADDPG 算法的智能体实现。
    """
    def __init__(self, agent_id, args):
        # 调用 BaseAgent 的构造函数
        super().__init__(agent_id, args)
        
        # MADDPG 策略的实例化
        self.policy = MADDPG(args, agent_id)
        # 增加一个名称，方便保存/加载
        self.agent_name = f'agent_{agent_id}' 

    def select_action(self, o, noise_rate, epsilon):
        # 保持与原始代码相似的风格和逻辑
        if np.random.uniform() < epsilon:
            # 探索：随机动作
            u = np.random.uniform(-self.args['high_action'], self.args['high_action'], self.args['action_shape'][self.agent_id])
        else:
            # 利用：策略网络输出动作
            inputs = torch.from_numpy(o).float().unsqueeze(0).to(self.args['device'])
            # 策略网络前向传播
            pi = self.policy.actor_network(inputs).squeeze(0)
            
            u = pi.cpu().numpy()
            
            # (注释掉的代码)
            # noise = noise_rate * self.args['high_action'] * np.random.randn(*u.shape)  # gaussian noise
            # u += noise
            
            # 动作裁剪
            u = np.clip(u, -self.args['high_action'], self.args['high_action'])
        
        return u.copy()

    def learn(self, transitions, other_agents):
        # 学习逻辑委托给具体的 MADDPG policy
        self.policy.train(transitions, other_agents)

    def save_model(self, save_path, episode):
        """
        保存 MADDPG 智能体的 actor 和 critic 网络模型。
        :param save_path: 存储模型的根目录。
        :param episode: 当前的训练回合数。
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        # 定义保存文件名
        file_name = f'episode_{episode}_{self.agent_name}.pt'
        save_file = os.path.join(save_path, file_name)

        # 保存策略网络的参数
        torch.save({
            'actor_params': self.policy.actor_network.state_dict(),
            'critic_params': self.policy.critic_network.state_dict(),
        }, save_file)
        # print(f"Model saved to {save_file}") # 可以选择打印

    def load_model(self, load_path):
        """
        从给定路径加载智能体的 actor 和 critic 网络模型。
        :param load_path: 包含模型文件的路径。
        """
        if not os.path.exists(load_path):
            print(f"Warning: Model path not found at {load_path}. Skipping load.")
            return

        print(f"Loading model for {self.agent_name} from {load_path}")
        
        checkpoint = torch.load(load_path, map_location=self.args['device'])
        
        # 加载策略网络的参数
        self.policy.actor_network.load_state_dict(checkpoint['actor_params'])
        self.policy.critic_network.load_state_dict(checkpoint['critic_params'])
        
        # 将目标网络参数与主网络同步（因为目标网络通常不单独保存）
        self.policy.target_actor_network.load_state_dict(checkpoint['actor_params'])
        self.policy.target_critic_network.load_state_dict(checkpoint['critic_params'])
