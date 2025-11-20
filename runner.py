from common.agent import Agent
from common.replay_buffer import Buffer
import os, datetime
from tqdm import tqdm
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class Runner:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.noise = args['noise_rate']
        self.epsilon = args['epsilon']
        self.episode_limit = args['max_episode_len']
        self.agents = self._init_agents()
        # self.k = args['ensemble_k']
        # self.S = [Buffer(args) for _ in range(self.k)]
        self.buffer = Buffer(args)
        self.str_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_path = self.args['save_dir'] + '/' + self.args['scenario_name'] + '_' + self.str_time
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.eval_step = 0
    def _init_agents(self):
        agents = []
        for i in range(self.args['n_agents']):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents
    def run(self):
        # returns = []
        writer = SummaryWriter(self.args['log_dir'])
        for time_step in tqdm(range(self.args['time_steps'])):
            # reset the environment
            if time_step % self.episode_limit == 0:
                s = self.env.reset()
            u = []
            actions = []
            # k = np.random.randint(0, self.k) # 增加
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    # action = agent.select_action(s[agent_id], self.noise, self.epsilon, k)  #增加
                    action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                    u.append(action)
                    actions.append(action)
            for _ in range(self.args['n_agents'], self.args['n_players']):
                actions.append([0, np.random.rand()*2-1, 0, np.random.rand()*2-1, 0])
            s_next, r, done, _ = self.env.step(actions)
            # self.buffer[k].store_episode(s[:self.args['n_agents']], u, r[:self.args['n_agents']], s_next[:self.args['n_agents']]) # 增加
            self.buffer.store_episode(s[:self.args['n_agents']], u,
                                         r[:self.args['n_agents']],
                                         s_next[:self.args['n_agents']])
            s = s_next
            # if self.buffer[k].current_size >= self.args['batch_size']: # 修改
            if self.buffer.current_size >= self.args['batch_size']: # 修改
                transitions = self.buffer.sample(self.args['batch_size'])  # 修改
                # transitions = self.buffer[k].sample(self.args['batch_size'])  # 修改
                for agent in self.agents:
                    transitions_t = transitions.copy()
                    other_agents = self.agents.copy()
                    other_agents.remove(agent)
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






