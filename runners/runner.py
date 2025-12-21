import os
import time
import datetime
from copy import deepcopy

import numpy as np
import torch
from utils.config_tools import init_writter
from tqdm import tqdm
import setproctitle
from core.replay_buffer import Buffer


class Runner:

    def __init__(self, args, env):
        self.args = args
        self.env = env

        # ====== 基本超参数 ======
        self.noise = args.get('noise_rate', 0.1)
        self.epsilon = args.get('epsilon', 0.1)
        self.episode_limit = args.get('max_episode_len', 25)
        self.time_steps = args.get('time_steps', 1_000_000)
        self.n_agents = args['n_agents']
        self.n_players = args.get('n_players', self.n_agents)
        self.use_public_buffer = args.get('use_public_buffer', True)

        # 训练相关
        self.batch_size = args.get('batch_size', 1024)
        self.evaluate_rate = args.get('evaluate_rate', 5_000)
        self.evaluate_episodes = args.get('evaluate_episodes', 10)
        self.evaluate_episode_len = args.get('evaluate_episode_len', self.episode_limit)
        self.log_interval = args.get('log_interval', 5_000)
        self.save_interval = args.get('save_interval', 50_000)
        self.warmup_steps = args.get('warmup_steps', 0)  # 可选 warmup

        # 设备 & 随机种子
        self.seed = args.get('seed', 1)
        self.device = args.get('device', 'cpu')

        # ====== 日志与保存路径 ======
        self.model_dir = args.get('save_dir', None)
        self.log_file = args.get('log_dir', None)

        # TensorBoard
        self.writer = init_writter(args)

        # 设置进程名
        setproctitle.setproctitle("what-can-I-say?-only success!")

        # ====== 初始化智能体 & buffer ======
        self.agents = self._init_agents()

        if self.use_public_buffer:
            self.buffer = Buffer(args)
        else:
            self.buffer = [Buffer(args) for _ in range(self.n_agents)]

        # 统计相关
        self.eval_step = 0
        self.total_step = 0
        self.train_episode_rewards = []
        self._last_log_time = time.time()

    # ------------------------------------------------------------------
    # 初始化部分
    # -----------------------------------------------------------------

    def _init_agents(self):
        if self.args['algo_name'] == 'maddpg':
            from core.agents.MADDPGAgent import MADDPGAgent as Agent
        elif self.args['algo_name'] == 'matd3' or self.args['algo_name'] == 'maerl':
            from core.agents.MATD3Agent import MATD3Agent as Agent
        else:
            print('[Runner] raise not implemented algorithm agent classes.')
            raise NotImplementedError
        agents = []
        for i in range(self.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    # ------------------------------------------------------------------
    # 核心训练 loop
    # ------------------------------------------------------------------
    def run(self):
        
        start_time = time.time()
        self.total_step = 0

        # warmup
        if self.warmup_steps > 0:
            print(f"[Runner] Start warmup for {self.warmup_steps} steps...")
            self.warmup(self.warmup_steps)
            print("[Runner] Warmup finished, start training.")

        s = self.env.reset()
        episode_reward = 0.0
        episode_idx = 0

        pbar = tqdm(range(self.time_steps), desc="Training", ncols=120)
        for t in pbar:
            self.total_step += 1

            # ================== 1. 选择动作 ==================
            actions, u = self._select_actions(s)

            # 填充其它非学习玩家动作
            for _ in range(self.n_agents, self.n_players):
                actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])

            # ================== 2. 与环境交互 ==================
            s_next, r, done, info = self.env.step(actions)

            # 只统计第一个智能体的奖励，或者你可以按需要改成 r.sum()
            episode_reward += r[0]

            # ================== 3. 存储到 buffer ==================
            self._store_transition(s, u, r, s_next)

            s = s_next

            # ================== 4. 训练智能体 ==================
            self._maybe_train()

            # ================== 5. episode 完成的处理 ==================
            if (t + 1) % self.episode_limit == 0:
                self.train_episode_rewards.append(episode_reward)
                episode_idx += 1
                episode_reward = 0.0
                s = self.env.reset()

            # ================== 6. 定期评估 ==================
            if self.total_step % self.evaluate_rate == 0:
                eval_avg_reward = self.evaluate()
                self.writer.add_scalar('eval/avg_reward', eval_avg_reward, self.eval_step)
                self.eval_step += 1

                # self.console_log(
                #     self.total_step,
                #     eval_info={"eval_avg_reward": eval_avg_reward}
                # )

            # ================== 7. 定期打印训练信息 ==================
            if self.total_step % self.log_interval == 0 and len(self.train_episode_rewards) > 0:
                avg_train_rew = float(np.mean(self.train_episode_rewards[-10:]))
                self.writer.add_scalar('train/avg_reward_10ep', avg_train_rew, self.total_step)
                # self.console_log(
                #     self.total_step,
                #     train_info={"avg_train_reward_10ep": avg_train_rew}
                # )

            # ================== 8. 定期保存模型（如果你的 Agent 支持的话） ==================
            if self.total_step % self.save_interval == 0:
                self.save()

        # 训练结束
        total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        print(f"[Runner] Training finished, total steps = {self.total_step}, time = {total_time}")
        self.close()

    # ------------------------------------------------------------------
    # warmup：随机策略填充 buffer
    # ------------------------------------------------------------------
    def warmup(self, warmup_steps):
        s = self.env.reset()
        for _ in tqdm(range(warmup_steps), desc="Warmup", ncols=120):
            actions = []
            for _ in range(self.n_agents):
                # 简单随机策略，你也可以写成 env.action_space.sample()
                actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
            for _ in range(self.n_agents, self.n_players):
                actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])

            s_next, r, done, info = self.env.step(actions)
            # 这里的 u 就是 actions[:n_agents]
            self._store_transition(s, actions[:self.n_agents], r, s_next)
            s = s_next
            # 你可以按照 episode_limit 重置
            # 如果环境有 done 标志也可以用 done 重置

    # ------------------------------------------------------------------
    # 行为选择 & 存储 & 训练
    # ------------------------------------------------------------------
    def _select_actions(self, s):
        """
        使用当前智能体策略选 action
        返回：
        - actions: 所有智能体的动作列表（长度 = n_agents）
        - u: 只包含学习智能体的动作列表（长度 = n_agents）
        """
        u = []
        actions = []
        with torch.no_grad():
            for agent_id, agent in enumerate(self.agents):
                action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                u.append(action)
                actions.append(action)
        return actions, u

    def _store_transition(self, s, u, r, s_next):
        """
        把 (s, u, r, s_next) 存入 buffer
        和你原来的逻辑保持一致
        """
        if self.use_public_buffer:
            # 只存前 n_agents 的信息
            self.buffer.store_episode(
                s[:self.n_agents], u, r[:self.n_agents], s_next[:self.n_agents]
            )
        else:
            for i in range(self.n_agents):
                self.buffer[i].store_episode(
                    s[:self.n_agents], u, r[:self.n_agents], s_next[:self.n_agents]
                )

    def _maybe_train(self):
        """
        判断 buffer 是否足够，若足够则进行一次/多次训练
        """
        if self.use_public_buffer:
            if self.buffer.current_size < self.batch_size:
                return
            transitions = self.buffer.sample(self.batch_size)
            for agent in self.agents:
                transitions_t = deepcopy(transitions)
                other_agents = [a for a in self.agents if a is not agent]
                agent.learn(transitions_t, other_agents)
        else:
            for i in range(self.n_agents):
                if self.buffer[i].current_size < self.batch_size:
                    continue
                transitions = self.buffer[i].sample(self.batch_size)
                for agent in self.agents:
                    transitions_t = deepcopy(transitions)
                    other_agents = [a for a in self.agents if a is not agent]
                    agent.learn(transitions_t, other_agents)

    # ------------------------------------------------------------------
    # 评估 & 日志 & 保存
    # ------------------------------------------------------------------
    def evaluate(self):
        """
        多次评估 episode，返回平均回报
        """
        returns = []
        for _ in range(self.evaluate_episodes):
            s = self.env.reset()
            rewards = 0.0
            for _ in range(self.evaluate_episode_len):
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0, 0)  # 无噪声，纯策略
                        actions.append(action)
                for _ in range(self.n_agents, self.n_players):
                    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])

                s_next, r, done, info = self.env.step(actions)
                rewards += r[0]
                s = s_next
            returns.append(rewards)
        return float(sum(returns) / len(returns))

    def console_log(self, step, train_info=None, eval_info=None):
        """
        类似 world_model_runner.console_log：
        - 打印到屏幕
        - 写入 progress.txt
        """
        now = time.time()
        header = (
            f"\n******** step: {step}, "
            f"elapsed: {now - self._last_log_time:.1f}s, "
            f"total: {datetime.timedelta(seconds=int(now))} ********"
        )
        print(header)
        self.log_file.write(header + "\n")

        if train_info is not None:
            line = "train_info: " + ", ".join(
                [f"{k}: {v:.4f}" for k, v in train_info.items()]
            )
            print(line)
            self.log_file.write(line + "\n")

        if eval_info is not None:
            line = "eval_info: " + ", ".join(
                [f"{k}: {v:.4f}" for k, v in eval_info.items()]
            )
            print(line)
            self.log_file.write(line + "\n")

        self.log_file.flush()
        self._last_log_time = now

    def save(self):
        """
        保存模型：
        这里假设每个 Agent 有 save/load 方法，如果没有你可以先留空
        """
        # 可以根据需要添加 if not hasattr(agent, "save"): return
        for i, agent in enumerate(self.agents):
            if hasattr(agent, "save"):
                agent.save(self.model_dir, i)

    def close(self):
        """
        关闭环境与日志
        """
        self.env.close()
        self.writer.close()
        # self.log_file.close()
