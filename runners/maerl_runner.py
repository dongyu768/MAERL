import time
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy
from runners.runner import Runner  # 导入你原本的 Runner

class MAERLRunner(Runner):
    def __init__(self, args, env):
        super().__init__(args, env)
        
        # ERL 特有的参数
        self.pop_size = args.get('pop_size', 10)
        # 假设 ERL 是按代数训练，这里重新定义进度条的长度含义
        self.n_generations = args.get('n_generations', 1000)
        self.len_per_episode = args.get('max_episode_len', 25)

    def _init_agents(self):
        from core.agents.MAERLAgent import MAERLAgent 
        agents = []
        for i in range(self.n_agents):
            # 传入 pop_size 等参数
            agent = MAERLAgent(i, self.args) 
            agents.append(agent)
        return agents

    def run(self):
        """
        重写核心训练循环: ERL 的流程是 评估种群 -> 收集梯度数据 -> 进化
        """
        start_time = time.time()
        self.total_step = 0

        # Warmup (复用父类)
        if self.warmup_steps > 0:
            self.warmup(self.warmup_steps)

        # === 进化主循环 (Generation Loop) ===
        pbar = tqdm(range(self.n_generations), desc="MAERL Training", ncols=120)
        
        for g in pbar:
            # ---------------------------------------------
            # 1. 种群评估 (Population Evaluation)
            # ---------------------------------------------
            # 我们需要分别为每个 agent 的种群中的每个个体计算 fitness
            # 结构：[Agent1_Fitness_List, Agent2_Fitness_List, ...]
            pop_fitness = [[] for _ in range(self.pop_size)]
            
            for pop_idx in range(self.pop_size):
                # 对种群中的第 pop_idx 个个体跑一次完整的 episode
                s = self.env.reset()
                episode_reward = np.zeros(self.n_agents)
                
                for step in range(self.len_per_episode):
                    self.total_step += 1
                    actions = []
                    # 获取动作：显式告诉 Agent 使用种群中的哪一个网络
                    with torch.no_grad():
                        for i, agent in enumerate(self.agents):
                            # MAERLAgent 需要实现带 pop_idx 的 select_action
                            action = agent.select_action(s[i], pop_idx=pop_idx)
                            actions.append(action)
                    
                    # 填充非学习玩家动作
                    for _ in range(self.n_agents, self.n_players):
                        actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])

                    s_next, r, done, info = self.env.step(actions)
                    episode_reward += r[:self.n_agents] # 累加奖励作为 Fitness
                    
                    # --- 关键点：进化的数据也可以用来训练 RL ---
                    # 我们可以把评估过程中的数据存入 Replay Buffer
                    # 注意：这里 u (actions) 直接存即可
                    self._store_transition(s, actions[:self.n_agents], r, s_next)
                    
                    s = s_next
                    if all(done): break
                
                # 记录该个体的 fitness
                for i in range(self.n_agents):
                    pop_fitness[i].append(episode_reward[i])

            # ---------------------------------------------
            # 2. 梯度强化学习更新 (RL Update)
            # ---------------------------------------------
            # 利用刚刚收集的数据更新 RL 部分 (Actor/Critic)
            # 可以根据配置决定更新多少次
            self._maybe_train() 

            # ---------------------------------------------
            # 3. 进化步骤 (Evolution Step)
            # ---------------------------------------------
            for i, agent in enumerate(self.agents):
                # 调用 agent 的进化函数，传入刚才评估得到的 fitness 列表
                # 这一步会进行：精英保留、交叉、变异、并将 RL 参数注入种群
                best_score = max(pop_fitness[i])
                agent.evolve(pop_fitness[i])
                
                # 记录日志
                self.writer.add_scalar(f'agent_{i}/best_fitness', best_score, g)

            # ---------------------------------------------
            # 4. 常规评估与保存 (复用父类逻辑)
            # ---------------------------------------------
            # 这里的 evaluate 使用的是 agent.rl_algo (RL策略) 或者是 种群中最好的策略
            if g % (self.evaluate_rate // self.len_per_episode) == 0: 
                # 注意：evaluate_rate 原本是 step，这里大概换算一下
                avg_reward = self.evaluate()
                self.writer.add_scalar('train/avg_reward_eval', avg_reward, self.total_step)
                # self.console_log(self.total_step, eval_info={"eval_avg_reward": avg_reward})
            
            if g % 100 == 0: # 定期保存
                self.save()

        print(f"[MAERL Runner] Finished.")
        self.close()
