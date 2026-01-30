'''Defined Various Agents Classes'''
import numpy as np
import torch
import os
import copy
from core.agents.BaseAgent import BaseAgent
from algorithms.matd3.matd3 import MATD3 
from algorithms.merl.neuroevolution import SSNE

class MAERLAgent(BaseAgent):
    """
    MAERL (Multi-Agent Evolutionary Reinforcement Learning) 智能体实现。
    集成了 梯度策略 (RL) 和 进化策略 (EA)。
    """
    def __init__(self, agent_id, args):
        super().__init__(agent_id, args)
        self.agent_name = f'agent_{agent_id}'
        self.pop_size = args.get('pop_size', 10)

        # 1. 初始化基于梯度的 RL 算法 (作为主要学习者和种群的引导者)
        self.rl_agent = MATD3(args, agent_id)

        # 2. 初始化进化模块
        self.evolver = SSNE(args)

        # 3. 初始化种群
        # 直接克隆 RL Agent 的 actor 网络结构来创建种群
        # 这样可以保证网络结构一致，且无需重复定义 MultiHeadActor
        self.population = []
        for _ in range(self.pop_size):
            # deepcopy 确保完全独立的参数副本
            net = copy.deepcopy(self.rl_agent.actor_network)
            net.eval() # 种群网络通常不需要在反向传播中计算梯度
            self.population.append(net)

    def select_action(self, o, noise_rate, epsilon, pop_idx=None):
        """
        选择动作。
        :param pop_idx: 如果不为 None，表示使用种群中的第 pop_idx 个个体进行决策（用于进化评估）。
                        如果为 None，表示使用 RL Agent 进行决策（用于梯度更新采样或最终测试）。
        """
        # 将观测转换为 Tensor
        inputs = torch.from_numpy(o).float().unsqueeze(0).to(self.args['device'])

        if pop_idx is not None:
            # === 进化模式 ===
            # 使用种群中的指定网络
            with torch.no_grad():
                pi = self.population[pop_idx](inputs).squeeze(0)
                u = pi.cpu().numpy()
                # 进化通常使用确定性策略，不加噪声，或者噪声已包含在参数变异中
        else:
            # === RL 模式 ===
            # 使用梯度更新的主网络 (与 MADDPGAgent 逻辑一致)
            if np.random.uniform() < epsilon:
                u = np.random.uniform(-self.args['high_action'], self.args['high_action'], self.args['action_shape'][self.agent_id])
            else:
                with torch.no_grad():
                    pi = self.rl_agent.actor_network(inputs).squeeze(0)
                    u = pi.cpu().numpy()
                    # 可以在这里加噪声，取决于具体的算法实现
                    # noise = noise_rate * self.args['high_action'] * np.random.randn(*u.shape)
                    # u += noise
        
        # 动作裁剪
        u = np.clip(u, -self.args['high_action'], self.args['high_action'])
        return u.copy()

    def learn(self, transitions, other_agents):
        """
        基于梯度的学习步骤 (RL Update)。
        Runner 会收集数据并传入这里。
        """
        # 将 learn 调用委托给内部的 matd3 实例
        # 注意：这里我们假设 other_agents 传入的是 MAERLAgent 列表
        # 但底层的 matd3.train 可能需要 accessing other_agents.policy
        # 我们需要做一个简单的转换，或者确保 matd3 能够处理
        
        # 提取其他智能体的内部 rl_policy (为了兼容底层的 train 函数)
        other_rl_policies = [agent.rl_agent for agent in other_agents]
        
        self.rl_agent.train(transitions, other_rl_policies)

    def evolve(self, fitness_list):
        """
        执行进化步骤 (Evolutionary Step)。
        在 Runner 完成一轮种群评估后调用。
        :param fitness_list: 对应 population 中每个个体的适应度分数列表
        """
        # 1. 进化算法核心步骤 (选择、变异、交叉)
        # 注意：SSNE 需要修改网络参数，传入 population 列表
        # 我们还需要传入 rl_agent.actor_network 用于将学到的梯度知识注入种群 (Lamarckian/Baldwinian Transfer)
        
        self.evolver.evolve(
            population=self.population, 
            fitness_scores=fitness_list, 
            rl_agent_net=self.rl_agent.actor_network # 将 RL 的网络传入用于注入
        )

        # (可选) 将进化产生的最优个体同步回 RL agent？
        # 通常 MAERL 是双向流动的：RL -> Pop (注入), Pop -> RL (作为 Buffer 数据来源 或 直接参数覆盖)
        # 如果你的策略需要将最优进化个体赋值给 RL agent，可以在这里做：
        # best_idx = np.argmax(fitness_list)
        # utils.hard_update(self.rl_agent.actor_network, self.population[best_idx])

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
        保存模型。主要保存 RL Agent 的参数，因为它是最终输出的策略。
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        file_name = f'episode_{episode}_{self.agent_name}.pt'
        save_file = os.path.join(save_path, file_name)

        # 保存 rl_agent 的网络
        torch.save({
            'actor_params': self.rl_agent.actor_network.state_dict(),
            'critic_params': self.rl_agent.critic_network.state_dict(),
            # 如果需要恢复训练，建议也保存种群，但这会让文件很大
            # 'population': [net.state_dict() for net in self.population] 
        }, save_file)

    def load_model(self, load_path):
        """
        加载模型。
        """
        if not os.path.exists(load_path):
            print(f"Warning: Model path not found at {load_path}")
            return

        print(f"Loading model for {self.agent_name} from {load_path}")
        checkpoint = torch.load(load_path, map_location=self.args['device'])
        
        # 恢复 RL Agent
        self.rl_agent.actor_network.load_state_dict(checkpoint['actor_params'])
        self.rl_agent.critic_network.load_state_dict(checkpoint['critic_params'])
        
        # 同步 Target 网络
        if hasattr(self.rl_agent, 'target_actor_network'):
            self.rl_agent.target_actor_network.load_state_dict(checkpoint['actor_params'])
            self.rl_agent.target_critic_network.load_state_dict(checkpoint['critic_params'])
            
        # 重新初始化种群以匹配加载的策略 (相当于所有种群从这个预训练模型开始进化)
        for net in self.population:
            net.load_state_dict(checkpoint['actor_params'])
