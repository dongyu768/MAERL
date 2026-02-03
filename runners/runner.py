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
from moviepy.editor import ImageSequenceClip


class Runner:

    def __init__(self, args, env):
        self.args = args
        self.env = env

        self.noise = args.get('noise_rate', 0.1)
        self.epsilon = args.get('epsilon', 0.1)
        self.episode_limit = args.get('max_episode_len', 25)
        self.time_steps = args.get('time_steps', 1_000_000)
        self.n_agents = args['n_agents']
        self.n_players = args.get('n_players', self.n_agents)
        self.use_public_buffer = args.get('use_public_buffer', True)

        self.batch_size = args.get('batch_size', 1024)
        self.evaluate_rate = args.get('evaluate_rate', 5_000)
        self.evaluate_episodes = args.get('evaluate_episodes', 10)
        self.evaluate_episode_len = args.get('evaluate_episode_len', self.episode_limit)
        self.log_interval = args.get('log_interval', 5_000)
        self.save_interval = args.get('save_interval', 50_000)
        self.warmup_steps = args.get('warmup_steps', 0)

        self.seed = args.get('seed', 1)
        self.device = args.get('device', 'cpu')

        self.model_dir = args.get('save_dir', None)
        self.log_file = args.get('log_dir', None)

        # TensorBoard
        # self.writer = init_writter(args)

        setproctitle.setproctitle("what-can-I-say?-only success!")

        self.agents = self.init_agents()

        if self.use_public_buffer:
            self.buffer = Buffer(args)
        else:
            self.buffer = [Buffer(args) for _ in range(self.n_agents)]

        self.eval_step = 0
        self.total_step = 0
        self.train_episode_rewards = []
        self._last_log_time = time.time()

    def init_agents(self):
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

    def run(self):
        self.writer = init_writter(self.args)
        start_time = time.time()
        self.total_step = 0
        if self.warmup_steps > 0:
            print(f"[Runner] Start warmup for {self.warmup_steps} steps...")
            self.warmup(self.warmup_steps)
            print("[Runner] Warmup finished, start training.")

        s = self.env.reset()
        episode_reward = 0.0
        episode_idx = 0

        # pbar = tqdm(range(self.time_steps), desc="Training", ncols=120)
        for t in tqdm(range(self.time_steps)):
            self.total_step += 1
            actions, u = self.select_actions(s)
            for _ in range(self.n_agents, self.n_players):
                actions.append(self.env.sample())

            s_next, r, done, info = self.env.step(actions)
            # episode_reward += r[0]
            good_reward = sum(r[:self.n_agents])
            episode_reward += good_reward

            self.store_transition(s, u, good_reward, s_next)
            s = s_next

            self.train()

            if (t + 1) % self.episode_limit == 0 or any(done):
                self.train_episode_rewards.append(episode_reward)
                episode_idx += 1
                episode_reward = 0.0
                s = self.env.reset()

            if self.total_step % self.evaluate_rate == 0:
                eval_g_reward, eval_adv_reward = self.evaluate()
                self.writer.add_scalar('eval/avg_good_reward', eval_g_reward, self.eval_step)
                self.writer.add_scalar('eval/avg_adv_reward', eval_adv_reward, self.eval_step)
                self.eval_step += 1
            if self.total_step % self.log_interval == 0 and len(self.train_episode_rewards) > 0:
                avg_train_rew = float(np.mean(self.train_episode_rewards[-10:]))
                self.writer.add_scalar('train/avg_reward_10ep', avg_train_rew, self.total_step)
            if self.total_step % self.save_interval == 0:
                self.save()
        total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        print(f"[Runner] Training finished, total steps = {self.total_step}, time = {total_time}")
        self.close()

    def warmup(self, warmup_steps):
        s = self.env.reset()
        for _ in tqdm(range(warmup_steps), desc="Warmup", ncols=120):
            actions = []
            for _ in range(self.n_agents):
                actions.append(self.env.sample())
            for _ in range(self.n_agents, self.n_players):
                actions.append(self.env.sample())

            s_next, r, done, _ = self.env.step(actions)
            self.store_transition(s, actions[:self.n_agents], sum(r[:self.n_agents]), s_next)
            s = s_next


    def select_actions(self, s):
        
        u = []
        actions = []
        with torch.no_grad():
            for agent_id, agent in enumerate(self.agents):
                action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                u.append(action)
                actions.append(action)
        return actions, u

    def store_transition(self, s, u, r, s_next):
        if self.use_public_buffer:
            # 只存前 n_agents 的信息
            self.buffer.store_episode(s[:self.n_agents], u, r, s_next[:self.n_agents])
        else:
            for i in range(self.n_agents):
                self.buffer[i].store_episode(s[:self.n_agents], u, r, s_next[:self.n_agents])

    def train(self):
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

    def evaluate(self):
        if self.args.get("evaluate", False):
            model_dir = os.path.join(self.args["eval_path"], "models/")
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"model dir not found: {model_dir}")
            for i in range(self.n_agents):
                self.agents[i].load(str(model_dir))
        g_returns = []
        a_returns = []
        for time_step in range(self.evaluate_episodes):
            s = self.env.reset()
            g_rewards = 0.0
            adv_rewards = 0.0
            frames = []
            for _ in range(self.evaluate_episode_len):
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0, 0)  # 无噪声，纯策略
                        actions.append(action)
                for _ in range(self.n_agents, self.n_players):
                    actions.append(self.env.sample())

                s_next, r, done, info = self.env.step(actions)
                print(f'reward:{r}')
                # if time_step == self.config['evaluate_episodes']-1:
                if self.args.get("evaluate", False):
                    # 保存图片
                    frame = self.env.render(mode='rgb_array')
                    frames.append(frame)
                good_reward = sum(r[:self.n_agents])
                g_rewards += good_reward
                adversary_reward = sum(r[self.n_agents:self.n_players])
                adv_rewards += adversary_reward
                s = s_next
            # if len(frames) > 0 and time_step == self.config['evaluate_episodes']-1:
            if self.args.get("evaluate", False):
                self.save_vidio(frames)
            g_returns.append(g_rewards)
            a_returns.append(adv_rewards)
        return float(sum(g_returns) / len(g_returns)), float(sum(a_returns) / len(a_returns))

    def save_vidio(self, frames):
        clip = ImageSequenceClip(frames, fps=30) 
        clip.write_videofile(self.args['vedio_dir'] + "record.mp4", codec="libx264")

    def save(self):
        for i, agent in enumerate(self.agents):
            if hasattr(agent, "save"):
                agent.save()

    def close(self):
        self.env.close()
        self.writer.close()
