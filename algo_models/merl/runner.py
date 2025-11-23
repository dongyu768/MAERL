from algo_models.maddpg.agent import EvolutionaryAgent
from common.replay_buffer import Buffer
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import Process, Pipe
from algo_models.merl.rollout_worker import rollout_worker

RANDOM_BASELINE = False

class Runner:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.agents = self._init_agents()
        self.buffer = [Buffer(args) for _ in range(args['n_agents'])]
        self.popn = [agent.popn for agent in self.agents]
        self.rollouts = [agent.rollout_actor for agent in self.agents]

        if self.args['popn_size'] > 0:
            self.evo_task_pipes = [Pipe() for _ in range(self.args['popn_size'])]
            self.evo_result_pipes = [Pipe() for _ in range(self.args['popn_size'])]
            self.evo_workers = [Process(target=rollout_worker, args=(self.args, i, 'evo', self.evo_task_pipes[i][1], self.evo_result_pipes[i][0],
				self.buffer_bucket, self.popn_bucket, True, RANDOM_BASELINE)) for i in range(self.args['popn_size'])]
            for worker in self.evo_workers: 
                worker.start()

        self.eval_step = 0
    
    def _init_agents(self):
        agents = []
        for i in range(self.args['n_agents']):
            agent = EvolutionaryAgent(i, self.args)
            agents.append(agent)        
        return agents
    
    def run(self):
        # returns = []
        writer = SummaryWriter(self.args['log_dir'])
        for time_step in tqdm(range(self.args['time_steps'])):
            # 重置环境
            if time_step % self.episode_limit == 0:
                s = self.env.reset()
            # 生成动作
            u = []  # 存储需要训练智能体的动作
            actions = []  # 存储所有玩家的动作，非智能体玩家的动作是随机生成的
            # k = np.random.randint(0, self.k) # 增加
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    # action = agent.select_action(s[agent_id], self.noise, self.epsilon, k)  #增加
                    action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                    u.append(action)
                    actions.append(action)
            for _ in range(self.args['n_agents'], self.args['n_players']):
                actions.append([0, np.random.rand()*2-1, 0, np.random.rand()*2-1, 0])
            # 与环境交互
            s_next, r, _, _ = self.env.step(actions)
            # 存储交互数据
            if self.args['use_public_buffer']:
                # self.buffer[k].store_episode(s[:self.args['n_agents']], u, r[:self.args['n_agents']], s_next[:self.args['n_agents']]) # 增加
                self.buffer.store_episode(s[:self.args['n_agents']], u, r[:self.args['n_agents']], s_next[:self.args['n_agents']])
            else:
                # 每个智能体自带buffer，不使用公共的buffer，需要在agent中定义
                for i in range(self.args['n_agents']):
                    self.buffer[i].store_episode(s[:self.args['n_agents']], u, r[:self.args['n_agents']], s_next[:self.args['n_agents']])
            s = s_next
            # if self.buffer[k].current_size >= self.args['batch_size']: # 修改
            if self.args['use_public_buffer']:
                if self.buffer.current_size >= self.args['batch_size']: # 修改
                    # 从buffer中采样用于训练
                    transitions = self.buffer.sample(self.args['batch_size'])  # 修改
                    # transitions = self.buffer[k].sample(self.args['batch_size'])  # 修改
                    for agent in self.agents:
                        transitions_t = transitions.copy()
                        other_agents = self.agents.copy()
                        other_agents.remove(agent)
                        # 每个智能体依次开始学习策略
                        agent.learn(transitions_t, other_agents)
                        # agent.learn(transitions, other_agents, k) # 增加
            else:
                # 每个智能体只使用自己缓存区的数据训练自己的策略
                for i in range(self.args['n_agents']):
                    if self.buffer[i].current_size >= self.args['batch_size']: # 修改
                    # 从buffer中采样用于训练
                        transitions = self.buffer[i].sample(self.args['batch_size'])  # 修改
                        # transitions = self.buffer[k].sample(self.args['batch_size'])  # 修改
                        for agent in self.agents:
                            transitions_t = transitions.copy()
                            other_agents = self.agents.copy()
                            other_agents.remove(agent)
                            # 每个智能体依次开始学习策略
                            agent.learn(transitions_t, other_agents)
                            # agent.learn(transitions, other_agents, k) # 增加
            if time_step > 0 and time_step % self.args['evaluate_rate'] == 0:
                eval_avg_reward = self.evaluate()
                writer.add_scalar('eval_avg_reward', eval_avg_reward, self.eval_step)
                self.eval_step += 1
            # self.noise = max(0.05, self.noise-0.0000005)
            # self.epsilon = max(0.05, self.epsilon - 0.0000005)
            # np.save(self.save_path + '/returns.pkl', returns)

    def evaluate(self):
        returns = []
        for _ in range(self.args['evaluate_episodes']):
            #reset the environment
            s = self.env.reset()
            rewards = 0
            for _ in range(self.args['evaluate_episode_len']):
                # self.env.render()
                actions = []
                # k = np.random.randint(0, self.k)  # 增加
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0, 0)
                        # action = agent.select_action(s[agent_id], 0, 0, k)# 增加
                        actions.append(action)
                for i in range(self.args['n_agents'], self.args['n_players']):
                    actions.append([0, np.random.rand()*2-1, 0, np.random.rand()*2-1, 0])
                s_next, r, done, info = self.env.step(actions)
                rewards += r[0]
                s = s_next
            returns.append(rewards)
            # print('Return is:', rewards)
        return sum(returns) / self.args['evaluate_episodes']






