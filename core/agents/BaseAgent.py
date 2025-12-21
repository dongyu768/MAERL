from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    智能体的抽象基类 (Abstract Base Class)。
    定义了所有智能体必须实现的接口，以确保可扩展性。
    """
    def __init__(self, agent_id, args):
        self.agent_id = agent_id
        self.args = args
        self.policy = None  # 具体策略实例留给子类实现

    @abstractmethod
    def select_action(self, observation, noise_rate, epsilon):
        """
        根据当前观测选择动作，考虑探索噪声和epsilon-greedy策略。
        
        :param observation: 当前观测值。
        :param noise_rate: 动作噪声率。
        :param epsilon: epsilon-greedy的epsilon值。
        :return: 智能体的动作。
        """
        pass

    @abstractmethod
    def learn(self, transitions, other_agents):
        """
        使用经验数据进行策略学习和网络更新。
        
        :param transitions: 从经验回放缓冲区采样的批次数据。
        :param other_agents: 环境中的其他智能体实例列表，用于协同学习（如MADDPG）。
        """
        pass
    
    @abstractmethod
    def save_model(self, save_path, episode):
        """保存智能体的策略网络和相关状态"""
        pass

    @abstractmethod
    def load_model(self, load_path):
        """从给定路径加载智能体的策略网络和相关状态"""
        pass
