'''
3架无人机参与博弈:
    1个对抗者与2个合作方
    若干静态地标(默认等于好方数量)
    每回合随机挑选1个目标地标 颜色标绿 其余为灰色
    合作方目标 尽快接近并占领目标地标 同时尽量让对抗者远离该地标
    对抗者目标 尽快接近目标地标以干扰抢占
观测：
    返回形状为(NUM_DRONES, OBS_DIM)的矩阵，便于集中式训练或简单多智能体算法对接 也可通过外部wrapper转为pettingzoo/RLlib格式
动作:
    默认VEL 每架无人机4维 前3维为方向 第4维为速度尺度 [0, 1] 内部限幅到SPEED_LIMIT
奖励:
    好方: 最小好方到目标地标的距离越小越好 同时鼓励对抗者远离
    坏方: 到目标地标的距离越小越好
终止与截断:
    默认无终止 按固定时长EPISODE_LEN_SEC截断
    越界或过大倾斜可触发截断 阈值可调整
'''
# envs/multidrone/MultiAdversaryAviary.py
import numpy as np
import pybullet as p
from gymnasium import spaces

from envs.drone_domain.BaseRLAviary import BaseRLAviary
from envs.drone_domain.drone_utils.enums import DroneModel, Physics, ActionType, ObservationType
from envs.drone_domain.control.DSLPIDControl import DSLPIDControl  # typical location control


class Scenario(BaseRLAviary):
    """
    Multi-agent strategic environment: 2 good drones vs 1 adversarial drone to capture a landmark.

    - Agents: first num_adversaries indices are adversaries (by default 0), the rest are good agents.
    - Landmarks: static spheres; one is randomly chosen as the goal per episode (colored green).
    - Observation (per-agent):
        [goal_rel(3, good only; adversary gets zeros),
         all_landmarks_rel(3*N_LM),
         other_agents_rel(3*(N-1))]
      Stacked to shape (NUM_DRONES, OBS_DIM).
    - Actions:
        VEL (default): [dx, dy, dz, speed_scale] in [-1, 1], mapped to velocity command.
        PID: [dx, dy, dz] waypoint increment per step.
        RPM: [r1, r2, r3, r4] normalized in [-1, 1] -> motor rpm.
    - Reward:
        Good agents: - min_good_dist_to_goal + lambda_adv * (sum adversary_dist_to_goal).
        Adversary:  - squared_dist_to_goal.
      The environment returns a single scalar reward (sum over per-agent rewards) for centralized training.
      Per-agent rewards are reported in info['per_agent_reward'].
    """
    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 3,
                 num_adversaries: int = 1,
                 num_landmarks: int = 3,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui: bool = False,
                 record: bool = False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.VEL,
                 episode_len_sec: float = 10.0,
                 arena_radius: float = 1.5,
                 landmark_radius: float = 0.08,
                 control_radius_scale: float = 2.0,
                 adv_distance_weight: float = 1.0,
                 occupancy_bonus_per_step: float = 0.02,
                 seed: int = 2026):
        assert num_drones >= 2 and num_adversaries >= 1 and num_adversaries < num_drones, \
            "Need at least 1 adversary and 1 good agent"
        self.NUM_ADVERSARIES = int(num_adversaries)
        self.NUM_GOOD = int(num_drones - num_adversaries)
        self.ADV_INDICES = list(range(self.NUM_ADVERSARIES))
        self.GOOD_INDICES = list(range(self.NUM_ADVERSARIES, num_drones))

        self.NUM_LANDMARKS = int(num_landmarks) if num_landmarks is not None else self.NUM_GOOD
        self.EPISODE_LEN_SEC = float(episode_len_sec)

        # Arena and landmark geometry
        self.ARENA_RADIUS = float(arena_radius)
        self.LM_RADIUS = float(landmark_radius)
        self.CONTROL_RADIUS = control_radius_scale * self.LM_RADIUS  # occupation radius

        # Reward shaping params
        self.LAMBDA_ADV = float(adv_distance_weight)
        self.OCCUPANCY_BONUS = float(occupancy_bonus_per_step)

        # Runtime buffers
        self.landmark_ids: list[int] = []
        self.landmark_pos: np.ndarray | None = None  # (NUM_LANDMARKS, 3)
        self.goal_landmark_idx: int = 0
        self.good_hold_steps: int = 0
        self.adv_hold_steps: int = 0

        # High-level controllers (for VEL/PID)
        self.ctrl = None
        self.SPEED_LIMIT = None  # m/s, set after super()

        # Seed
        if seed is not None:
            np.random.seed(seed)

        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act)

        # Speed limit for velocity actions (~3% of max speed)
        try:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000.0 / 3600.0)
        except Exception:
            self.SPEED_LIMIT = 1.0  # fallback

        # Create controllers if needed
        if self.ACT_TYPE in (ActionType.VEL, ActionType.PID):
            if DSLPIDControl is None:
                raise ImportError("DSLPIDControl not found. Please adjust import path or provide a controller.")
            self.ctrl = [DSLPIDControl(self.DRONE_MODEL) for _ in range(self.NUM_DRONES)]

    # ---------- Spaces ----------

    def _actionSpace(self):
        if self.ACT_TYPE == ActionType.RPM:
            ad = 4
        elif self.ACT_TYPE == ActionType.PID:
            ad = 3
        elif self.ACT_TYPE == ActionType.VEL:
            ad = 4
        else:
            raise NotImplementedError(f"Unsupported action type {self.ACT_TYPE}")

        self.ACT_DIM = ad
        return spaces.Box(low=-1.0, high=1.0, shape=(self.NUM_DRONES, ad), dtype=np.float32)

    def _observationSpace(self):
        # Per-agent obs: goal_rel(3) + landmarks_rel(3*L) + others_rel(3*(N-1))
        self.OBS_DIM = 3 + 3 * self.NUM_LANDMARKS + 3 * (self.NUM_DRONES - 1)
        low = -np.inf * np.ones((self.NUM_DRONES, self.OBS_DIM), dtype=np.float32)
        high = np.inf * np.ones((self.NUM_DRONES, self.OBS_DIM), dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    # ---------- Entities ----------

    def _addObstacles(self):
        """Create landmark spheres and select a goal landmark."""
        self.landmark_ids = []
        self.landmark_pos = []
        # Uniformly scatter landmarks on a disc at fixed z
        for _ in range(self.NUM_LANDMARKS):
            theta = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0.2 * self.ARENA_RADIUS, 0.9 * self.ARENA_RADIUS)
            pos = np.array([r * np.cos(theta), r * np.sin(theta), 0.25], dtype=float)
            col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=self.LM_RADIUS, physicsClientId=self.CLIENT)
            vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=self.LM_RADIUS,
                                         rgbaColor=[0.15, 0.15, 0.15, 1.0], physicsClientId=self.CLIENT)
            body = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id,
                                     basePosition=pos.tolist(), physicsClientId=self.CLIENT)
            self.landmark_ids.append(body)
            self.landmark_pos.append(pos)

        self.landmark_pos = np.vstack(self.landmark_pos)
        # Pick a goal and color it green
        self.goal_landmark_idx = int(np.random.randint(0, self.NUM_LANDMARKS))
        p.changeVisualShape(self.landmark_ids[self.goal_landmark_idx], -1,
                            rgbaColor=[0.15, 0.65, 0.15, 1.0], physicsClientId=self.CLIENT)

        # Reset occupancy timers
        self.good_hold_steps = 0
        self.adv_hold_steps = 0

    # ---------- Core RL hooks ----------

    def _computeObs(self):
        # Ensure positions up to date
        # self.pos shape: (N, 3)
        N = self.NUM_DRONES
        L = self.NUM_LANDMARKS
        goal_pos = self.landmark_pos[self.goal_landmark_idx]

        obs = np.zeros((N, self.OBS_DIM), dtype=np.float32)
        for i in range(N):
            # Goal relative (3); adversary does not see it -> zeros
            goal_rel = goal_pos - self.pos[i]
            if i in self.ADV_INDICES:
                goal_rel = np.zeros(3, dtype=np.float32)

            # All landmarks relative
            lm_rel = (self.landmark_pos - self.pos[i]).reshape(3 * L)

            # Other agents relative
            others_rel = []
            for j in range(N):
                if j == i:
                    continue
                others_rel.append(self.pos[j] - self.pos[i])
            others_rel = np.array(others_rel, dtype=np.float32).reshape(3 * (N - 1))

            obs[i, :] = np.hstack([goal_rel, lm_rel, others_rel]).astype(np.float32)

        return obs

    def _preprocessAction(self, action):
        """
        Map high-level multi-agent action to per-motor rpm for all drones.
        action: np.ndarray (NUM_DRONES, ACT_DIM) in [-1, 1]
        """
        N = self.NUM_DRONES
        rpm = np.zeros((N, 4), dtype=np.float32)

        if self.ACT_TYPE == ActionType.RPM:
            # Direct normalized RPM around hover with +/- 5% band (can be adjusted)
            for i in range(N):
                base = self.HOVER_RPM
                rpm[i, :] = np.clip(base * (1.0 + 0.05 * action[i, :]), 0.0, self.MAX_RPM)
            return rpm

        assert self.ctrl is not None, "High-level control requires a controller (e.g., DSLPIDControl)."

        # Cached state
        # Per-drone: [pos3 quat4 rpy3 vel3 ang_v3 last_action4] -> we need pos, rpy, vel
        for i in range(N):
            state = self._getDroneStateVector(i)
            cur_pos = state[0:3]
            cur_rpy = state[7:10]  # roll, pitch, yaw
            cur_yaw = float(cur_rpy[2])

            if self.ACT_TYPE == ActionType.PID:
                # Interpret as small waypoint step per control step
                step_vec = 0.1 * np.clip(action[i, 0:3], -1.0, 1.0)  # 10 cm step
                target_pos = cur_pos + step_vec
                target_rpy = np.array([0.0, 0.0, cur_yaw])  # keep current yaw
                rpm[i, :], _, _ = self.ctrl[i].computeControlFromState(control_timestep=self.CTRL_TIMESTEP,
                                                                       state=state,
                                                                       target_pos=target_pos,
                                                                       target_rpy=target_rpy)
            elif self.ACT_TYPE == ActionType.VEL:
                # Directional velocity with magnitude from |a[3]|
                vec = action[i, 0:3].astype(float)
                mag = np.linalg.norm(vec) + 1e-6
                vec = vec / mag
                speed = float(np.clip(abs(action[i, 3]), 0.0, 1.0)) * float(self.SPEED_LIMIT)
                target_vel = vec * speed
                # Integrate 1 step ahead to build a local waypoint (works well with DSLPID)
                target_pos = cur_pos + target_vel * self.CTRL_TIMESTEP
                target_rpy = np.array([0.0, 0.0, cur_yaw])
                rpm[i, :], _, _ = self.ctrl[i].computeControlFromState(control_timestep=self.CTRL_TIMESTEP,
                                                                       state=state,
                                                                       target_pos=target_pos,
                                                                       target_rpy=target_rpy)
            else:
                raise NotImplementedError(f"Unsupported action type {self.ACT_TYPE}")

        # Safety clip
        return np.clip(rpm, 0.0, self.MAX_RPM)

    def _computeReward(self):
        """
        Centralized scalar reward for training; per-agent rewards reported in info.
        - Good agents: -min_dist(good -> goal) + lambda_adv * sum_dist(adv -> goal)
        - Adversary:   -||adv -> goal||^2
        + small occupancy bonus for holding the goal area while the opponent is away.
        """
        goal = self.landmark_pos[self.goal_landmark_idx]
        good_dists = [np.linalg.norm(self.pos[i] - goal) for i in self.GOOD_INDICES]
        adv_dists = [np.linalg.norm(self.pos[i] - goal) for i in self.ADV_INDICES]

        # Good agents reward (shared shaping)
        pos_rew = -np.min(good_dists) if len(good_dists) > 0 else 0.0
        adv_sep_rew = self.LAMBDA_ADV * np.sum(adv_dists) if len(adv_dists) > 0 else 0.0
        good_reward = pos_rew + adv_sep_rew

        # Adversary reward
        adv_reward = 0.0
        for d in adv_dists:
            adv_reward += - (d ** 2)

        # Occupancy bonus (who controls the goal area this step?)
        holder = self._who_controls_goal()
        occ_bonus = 0.0
        if holder == "good":
            self.good_hold_steps += 1
            occ_bonus += self.OCCUPANCY_BONUS
        elif holder == "adv":
            self.adv_hold_steps += 1
            occ_bonus -= self.OCCUPANCY_BONUS  # adversary holding reduces team reward

        # Aggregate scalar reward (centralized)
        total = good_reward + adv_reward + occ_bonus
        return float(total)

    def _computeTerminated(self):
        # No hard terminal by default; could add collisions/crash conditions here.
        return False

    def _computeTruncated(self):
        # Safety truncation: out of arena or excessive tilt, or episode timeout
        # Use cached state vector for tilt checks
        N = self.NUM_DRONES
        for i in range(N):
            s = self._getDroneStateVector(i)
            pos = s[0:3]
            roll, pitch = s[7], s[8]
            if (np.linalg.norm(pos[0:2]) > self.ARENA_RADIUS + 0.2) or (pos[2] < 0.0) or (pos[2] > 2.5):
                return True
            if abs(roll) > 0.6 or abs(pitch) > 0.6:
                return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeInfo(self):
        goal = self.landmark_pos[self.goal_landmark_idx]
        good_dists = [np.linalg.norm(self.pos[i] - goal) for i in self.GOOD_INDICES]
        adv_dists = [np.linalg.norm(self.pos[i] - goal) for i in self.ADV_INDICES]

        # Per-agent reward (for analysis or MA wrappers)
        per_agent_reward = np.zeros(self.NUM_DRONES, dtype=np.float32)
        # Good
        if len(good_dists) > 0:
            pos_rew = -np.min(good_dists)
            adv_sep = self.LAMBDA_ADV * np.sum(adv_dists)
            for gi in self.GOOD_INDICES:
                per_agent_reward[gi] = pos_rew + adv_sep
        # Adversary
        for ai, d in zip(self.ADV_INDICES, adv_dists):
            per_agent_reward[ai] = - (d ** 2)

        return {
            "goal_landmark": int(self.goal_landmark_idx),
            "good_min_dist": float(np.min(good_dists) if len(good_dists) else np.inf),
            "adv_dist": float(adv_dists[0] if len(adv_dists) else np.inf),
            "holder": self._who_controls_goal(),
            "good_hold_steps": int(self.good_hold_steps),
            "adv_hold_steps": int(self.adv_hold_steps),
            "per_agent_reward": per_agent_reward,
        }

    # ---------- Helpers ----------

    def _who_controls_goal(self):
        """Return 'good' if any good agent is within control radius and all adversaries are outside,
           'adv' if any adversary is within and all goods are outside, else 'none'."""
        goal = self.landmark_pos[self.goal_landmark_idx]
        good_in = any(np.linalg.norm(self.pos[i] - goal) <= self.CONTROL_RADIUS for i in self.GOOD_INDICES)
        adv_in = any(np.linalg.norm(self.pos[i] - goal) <= self.CONTROL_RADIUS for i in self.ADV_INDICES)
        if good_in and not adv_in:
            return "good"
        if adv_in and not good_in:
            return "adv"
        return "none"