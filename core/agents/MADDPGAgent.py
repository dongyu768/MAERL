'''Defined Various Agents Classes'''
from algorithms.maddpg.maddpg import MADDPG
from core.agents.BaseAgent import BaseAgent
import numpy as np
import torch
import os
from utils.env_tools import check
from typing import Optional

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
            inputs = check(o)
            inputs = inputs.to(self.args['device'])
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

    def save(self):
        """
        Save actor/critic of this agent to:
          {model_dir}/agent_{agent_id}/actor.pth
          {model_dir}/agent_{agent_id}/critic.pth
        """
        agent_dir = os.path.join(self.args['save_dir'], self.agent_name)
        os.makedirs(agent_dir, exist_ok=True)

        actor_path = os.path.join(agent_dir, "actor.pth")
        critic_path = os.path.join(agent_dir, "critic.pth")

        # 只保存权重，保持与旧产物一致
        torch.save(self.policy.actor_network.state_dict(), actor_path)
        torch.save(self.policy.critic_network.state_dict(), critic_path)

    def load(self, model_dir):
        """
        Load actor/critic weights from:
          {model_dir}/agent_{agent_id}/actor.pth
          {model_dir}/agent_{agent_id}/critic.pth
        """
        agent_dir = os.path.join(model_dir, self.agent_name)
        actor_path = os.path.join(agent_dir, "actor.pth")
        critic_path = os.path.join(agent_dir, "critic.pth")

        # 先加载到 CPU 以避免跨设备问题，随后再迁移到目标 device
        actor_sd = torch.load(actor_path, map_location="cpu")
        critic_sd = torch.load(critic_path, map_location="cpu")

        self.policy.actor_network.load_state_dict(actor_sd,strict=True)
        self.policy.critic_network.load_state_dict(critic_sd,strict=True)
        self.policy.actor_target_network.load_state_dict(actor_sd,strict=True)
        self.policy.critic_target_network.load_state_dict(critic_sd,strict=True)

        if self.args['device'] is not None:
            self.policy.actor_network.to(self.args['device'])
            self.policy.critic_network.to(self.args['device'])
            self.policy.actor_target_network.to(self.args['device'])
            self.policy.critic_target_network.to(self.args['device'])

        # 评估时默认 eval 模式
        self.policy.actor_network.eval()
        self.policy.critic_network.eval()

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
