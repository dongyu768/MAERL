# envs/drone_mpe/environment.py
import numpy as np
import gym
from gym import spaces


from envs.drone_domain.scenarios.multidrone_adversary import Scenario
from envs.drone_domain.drone_utils.enums import ActionType, ObservationType, DroneModel, Physics


class DroneMultiAgentEnv(gym.Env):
    """
    MPE-compatible wrapper over MultiAdversaryAviary (2 good vs 1 adversary).
    - step takes a list (or NxA array) of per-agent actions and returns per-agent obs/reward/done/info.
    - action_space/observation_space: lists with one entry per agent (MPE convention).
    - optional discrete_action=True maps 7 discrete actions to VEL actions.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 aviary_cls=Scenario,
                 aviary_kwargs=None,
                 discrete_action=False,
                 shared_reward=False):
        super().__init__()
        self.aviary_cls = aviary_cls
        self.aviary_kwargs = {} if aviary_kwargs is None else dict(aviary_kwargs)
        self.discrete_action = bool(discrete_action)
        self.shared_reward = bool(shared_reward)

        # Enforce a sensible default for policy-level control
        if 'act' not in self.aviary_kwargs:
            self.aviary_kwargs['act'] = ActionType.VEL
        if 'obs' not in self.aviary_kwargs:
            self.aviary_kwargs['obs'] = ObservationType.KIN
        # Build underlying aviary
        self.aviary = self.aviary_cls
        # self.aviary = self.aviary_cls(**self.aviary_kwargs)

        # Agents
        self.n = int(self.aviary.NUM_DRONES)
        self.agent_names = [f"agent_{i}" for i in range(self.n)]

        # Spaces per agent (MPE convention)
        self._build_spaces()

    # ---------- Spaces ----------

    def _build_spaces(self):
        # Observation: (OBS_DIM,) per agent
        if len(self.aviary.observation_space.shape) != 2:
            raise ValueError("Underlying aviary obs space must be (N, OBS_DIM).")
        obs_dim = int(self.aviary.observation_space.shape[1])
        self.observation_space = [spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
                                  for _ in range(self.n)]

        # Action: discrete 7 (noop, ±x, ±y, ±z) OR Box (ACT_DIM,)
        if self.discrete_action:
            self.action_space = [spaces.Discrete(7) for _ in range(self.n)]
            self._act_dim = 4 if self.aviary.ACT_TYPE == ActionType.VEL else 3
        else:
            if len(self.aviary.action_space.shape) != 2:
                raise ValueError("Underlying aviary action space must be (N, ACT_DIM).")
            act_dim = int(self.aviary.action_space.shape[1])
            self._act_dim = act_dim
            self.action_space = [spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
                                 for _ in range(self.n)]

    # ---------- API ----------

    def reset(self):
        # Gymnasium-style reset -> obs_mat, info; we return obs_n only (MPE convention)
        obs_mat, _ = self.aviary.reset()
        return self._split_obs(obs_mat)

    def step(self, action_n):
        """
        action_n: list of per-agent actions or an (N, A) array.
        Returns: obs_n, rew_n, done_n, info_n (MPE convention)
        """
        act_mat = self._merge_action(action_n)
        obs_mat, reward, terminated, truncated, info = self.aviary.step(act_mat)

        # Per-agent observations
        obs_n = self._split_obs(obs_mat)

        # Per-agent rewards: prefer info['per_agent_reward'], else share/replicate
        if isinstance(info, dict) and ('per_agent_reward' in info) and \
           (hasattr(info['per_agent_reward'], '__len__')) and len(info['per_agent_reward']) == self.n:
            rew_arr = np.asarray(info['per_agent_reward'], dtype=np.float32)
            if self.shared_reward:
                rew_arr = np.full(self.n, np.sum(rew_arr), dtype=np.float32)
            rew_n = [float(r) for r in rew_arr]
        else:
            # Fallback: aggregate to all or share
            if self.shared_reward:
                rew_n = [float(reward)] * self.n
            else:
                # If no per-agent reward available, split evenly as a neutral fallback
                rew_n = [float(reward) / self.n] * self.n

        # Per-agent done flags: replicate episode status
        done_flag = bool(terminated or truncated)
        done_n = [done_flag] * self.n

        # Per-agent infos: keep global info under the same dict for each agent
        info_n = {'n': [dict(info) if isinstance(info, dict) else {} for _ in range(self.n)]}

        return obs_n, rew_n, done_n, info_n

    def render(self, mode='human'):
        # Delegate to underlying aviary (GUI recommended)
        return self.aviary.render(mode=mode)

    def close(self):
        self.aviary.close()

    # ---------- Helpers ----------

    def _split_obs(self, obs_mat):
        """(N, D) -> list of D-vectors"""
        obs_mat = np.asarray(obs_mat, dtype=np.float32)
        if obs_mat.ndim != 2 or obs_mat.shape[0] != self.n:
            raise ValueError(f"Unexpected obs shape {obs_mat.shape}, expected (N, D) with N={self.n}")
        return [obs_mat[i].copy() for i in range(self.n)]

    def _merge_action(self, action_n):
        """List/array of per-agent actions -> (N, ACT_DIM) array in [-1,1]."""
        if self.discrete_action:
            # Map 7 discrete actions to VEL action (dx,dy,dz,speed_scale) or PID (dx,dy,dz)
            act_mat = np.zeros((self.n, self._act_dim), dtype=np.float32)
            for i, a in enumerate(action_n):
                a = int(a) if not isinstance(a, (np.ndarray, list)) else int(np.asarray(a).item())
                # 0: noop, 1:+x, 2:-x, 3:+y, 4:-y, 5:+z, 6:-z
                vec = np.zeros(3, dtype=np.float32)
                if a == 1:
                    vec[0] = +1.0
                elif a == 2:
                    vec[0] = -1.0
                elif a == 3:
                    vec[1] = +1.0
                elif a == 4:
                    vec[1] = -1.0
                elif a == 5:
                    vec[2] = +1.0
                elif a == 6:
                    vec[2] = -1.0
                if self.aviary.ACT_TYPE == ActionType.VEL:
                    # [dx,dy,dz, speed_scale]
                    act_mat[i, 0:3] = vec
                    act_mat[i, 3] = 1.0 if a != 0 else 0.0
                else:
                    # PID: 3D waypoint increment
                    act_mat[i, 0:3] = vec
            return act_mat

        # Continuous case: accept list or array
        arr = np.asarray(action_n, dtype=np.float32)
        if arr.ndim == 1 and arr.shape[0] == self._act_dim:
            arr = np.tile(arr, (self.n, 1))
        if arr.ndim != 2 or arr.shape[0] != self.n or arr.shape[1] != self._act_dim:
            raise ValueError(f"Bad action shape {arr.shape}, expected (N,{self._act_dim})")
        return np.clip(arr, -1.0, 1.0)
