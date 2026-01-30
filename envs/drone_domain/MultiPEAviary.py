# envs/drone_domain/YumingPursuitAviary.py
import math
from typing import Dict, List, Sequence

import numpy as np
import pybullet as p
from gymnasium import spaces

from envs.drone_domain.BaseRLAviary import BaseRLAviary
from envs.drone_domain.drone_utils.enums import (
    ActionType,
    DroneModel,
    ObservationType,
    Physics,
)


class MultiPEAviary(BaseRLAviary):
    """A drop-in Gymnasium env mirroring CI-HRL's CEFC task"""
    """
    Key pieces: 
        - configurable rewards/targets/predator speed,
        - overriden reset/step to manage adversary & target state,
        - LiDAR and neighbor‑consensus encoders inside _computeObs, and 
        - reward/termination logic faithful to the paper’s five-term structure
    """
    """Multi‑constraint pursuit–evasion task inspired by CI‑HRL (Yuming et al., 2025)."""

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        num_drones: int = 8,
        neighbourhood_radius: float = 3.0,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 30,
        gui: bool = False,
        record: bool = False,
        obs: ObservationType = ObservationType.KIN,
        act: ActionType = ActionType.RPM,
        num_targets: int = 2,
        communication_range: float = 3.0,
        target_radius: float = 2.0,
        episode_len_sec: float = 20.0,
        lidar_num_rays: int = 12,
        lidar_max_distance: float = 6.0,
        predator_speed: float = 1.0,
        predator_altitude: float = 0.8,
        target_decay: float = 0.01,
        safe_evasion_radius: float = 2.0,
        collision_margin: float = 0.2,
        reward_weights = None,
    ):
        self.EPISODE_LEN_SEC = episode_len_sec
        self.NUM_TARGETS = num_targets
        self.COMM_RADIUS = communication_range
        self.TARGET_RADIUS = target_radius
        self.LIDAR_RAYS = lidar_num_rays
        self.LIDAR_RANGE = lidar_max_distance
        self.PREDATOR_SPEED = predator_speed
        self.PREDATOR_ALT = predator_altitude
        self.TARGET_DECAY = target_decay
        self.SAFE_RADIUS = safe_evasion_radius
        self.COLLISION_MARGIN = collision_margin
        self.target_positions = np.zeros((self.NUM_TARGETS, 3), dtype=np.float32)
        self.target_urgency = np.ones(self.NUM_TARGETS, dtype=np.float32)
        self.lidar_angles = np.linspace(0.0, 2.0 * np.pi, self.LIDAR_RAYS, endpoint=False)
        self.predator_id = None
        self.predator_pos = np.zeros(3, dtype=np.float32)
        self.predator_vel = np.zeros(3, dtype=np.float32)
        self.collision_flag = False
        self.reward_weights = reward_weights or {
            "formation": 8.0,
            "navigation": 3.0,
            "task": 5.0,
            "evasion": 4.0,
            "collision": 6.0,
        }
        self._action_components = self._resolve_action_dim(act)
        self._action_hist_len = max(1, ctrl_freq // 2)
        self._consensus_dim = 6
        self._neighbor_slots = min(num_drones - 1, 4)
        self._obs_dim = self._calculate_obs_dim()
        self._last_contacts: List = []
        self.drone_target_assignment = np.zeros(num_drones, dtype=np.int32)
        self.lidar_cache = np.ones((num_drones, self.LIDAR_RAYS), dtype=np.float32)
        super().__init__(
            drone_model=drone_model,
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
            act=act,
        )
        self._target_markers: List[int] = []
        self._sample_targets()
        self._spawn_predator()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._sample_targets()
        self._reset_predator_pose()
        self._update_target_assignments()
        obs = self._computeObs()
        info = self._computeInfo()
        return obs, info

    def step(self, action):
        super().step(action)
        self._update_post_step_state()
        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()
        done = terminated or truncated
        return obs, reward, done, info

    # ------------------------- observations ------------------------- #

    # def _observationSpace(self):
    #     low = -np.inf * np.ones((self.NUM_DRONES, self._obs_dim), dtype=np.float32)
    #     high = np.inf * np.ones_like(low)
    #     return spaces.Box(low=low, high=high, dtype=np.float32)

    def _computeObs(self):
        self._update_lidar()
        obs = np.zeros((self.NUM_DRONES, self._obs_dim), dtype=np.float32)
        for i in range(self.NUM_DRONES):
            cursor = 0
            pos = self.pos[i]
            vel = self.vel[i]
            obs[i, cursor : cursor + 3] = pos
            cursor += 3
            obs[i, cursor : cursor + 3] = vel
            cursor += 3
            obs[i, cursor : cursor + 3] = self.predator_pos - pos
            cursor += 3
            obs[i, cursor : cursor + 3] = self.predator_vel - vel
            cursor += 3
            for t in range(self.NUM_TARGETS):
                rel = self.target_positions[t] - pos
                obs[i, cursor : cursor + 3] = rel
                obs[i, cursor + 3] = self.target_urgency[t]
                cursor += 4
            cursor += self._encode_neighbors(i, obs[i, cursor:])
            obs[i, cursor : cursor + self._consensus_dim] = self._consensus_encoding(i)
            cursor += self._consensus_dim
            obs[i, cursor : cursor + self.LIDAR_RAYS] = self.lidar_cache[i]
            cursor += self.LIDAR_RAYS
            hist = self._action_history_for(i)
            obs[i, cursor : cursor + hist.size] = hist
        return obs

    # ------------------------- rewards & termination ------------------------- #

    def _computeReward(self):
        formation_err = self._formation_error()
        nav_dist = self._navigation_cost()
        coverage = self._coverage_score()
        evasion = self._evasion_penalty()
        collision = float(self.collision_flag or np.any(self.pos[:, 2] < self.COLLISION_MARGIN))
        reward = (
            -self.reward_weights["formation"] * formation_err
            - self.reward_weights["navigation"] * nav_dist
            + self.reward_weights["task"] * coverage
            - self.reward_weights["evasion"] * evasion
            - self.reward_weights["collision"] * collision
        )
        return float(reward)

    def _computeTerminated(self):
        targets_done = np.all(self.target_urgency <= 1e-3)
        predator_hit = np.any(np.linalg.norm(self.pos - self.predator_pos, axis=1) < 0.2)
        return bool(targets_done or predator_hit or self.collision_flag)

    def _computeTruncated(self):
        timeout = (self.step_counter / self.PYB_FREQ) > self.EPISODE_LEN_SEC
        tilt = np.any(np.abs(self.rpy[:, :2]) > 0.6)
        return bool(timeout or tilt)

    def _computeInfo(self):
        info = {
            "target_urgency": self.target_urgency.copy(),
            "assignments": self.drone_target_assignment.copy(),
            "avg_formation_error": self._formation_error(),
            "collision": self.collision_flag,
        }
        return info

    # ------------------------- helpers ------------------------- #

    def _resolve_action_dim(self, act: ActionType) -> int:
        if act in (ActionType.RPM, ActionType.VEL):
            return 4
        if act == ActionType.PID:
            return 3
        return 1

    def _calculate_obs_dim(self) -> int:
        base = 3 + 3 + 3 + 3
        target_feat = self.NUM_TARGETS * 4
        neighbor_feat = self._neighbor_slots * 6
        lidar = self.LIDAR_RAYS
        history = self._action_hist_len * self._action_components
        return base + target_feat + neighbor_feat + self._consensus_dim + lidar + history

    def _action_history_for(self, drone_idx: int) -> np.ndarray:
        history: List[float] = []
        for past in self.action_buffer:
            history.extend(past[drone_idx, : self._action_components])
        return np.array(history, dtype=np.float32)

    def _encode_neighbors(self, idx: int, buffer: np.ndarray) -> int:
        neighbors = self._neighbors_of(idx)
        slot = 0
        encoded = 0
        for j in neighbors[: self._neighbor_slots]:
            rel_pos = self.pos[j] - self.pos[idx]
            rel_vel = self.vel[j] - self.vel[idx]
            buffer[slot : slot + 3] = rel_pos
            buffer[slot + 3 : slot + 6] = rel_vel
            slot += 6
            encoded += 6
        if encoded < self._neighbor_slots * 6:
            buffer[encoded : self._neighbor_slots * 6] = 0.0
        return self._neighbor_slots * 6

    def _consensus_encoding(self, idx: int) -> np.ndarray:
        neighbors = self._neighbors_of(idx)
        if not neighbors:
            return np.zeros(self._consensus_dim, dtype=np.float32)
        positions = self.pos[neighbors]
        velocities = self.vel[neighbors]
        mean_pos = np.mean(positions, axis=0)
        mean_vel = np.mean(velocities, axis=0)
        spread = np.max(np.linalg.norm(positions - mean_pos, axis=1))
        return np.hstack([mean_pos - self.pos[idx], mean_vel - self.vel[idx], spread]).astype(np.float32)

    def _neighbors_of(self, idx: int) -> List[int]:
        neigh = []
        for j in range(self.NUM_DRONES):
            if j == idx:
                continue
            if np.linalg.norm(self.pos[j] - self.pos[idx]) <= self.COMM_RADIUS:
                neigh.append(j)
        return neigh

    def _update_lidar(self):
        origins: List[Sequence[float]] = []
        ends: List[Sequence[float]] = []
        for i in range(self.NUM_DRONES):
            start = self.pos[i] + np.array([0.0, 0.0, self.L])
            for angle in self.lidar_angles:
                direction = np.array([math.cos(angle), math.sin(angle), 0.0])
                origins.append(start)
                ends.append(start + direction * self.LIDAR_RANGE)
        results = p.rayTestBatch(origins, ends, physicsClientId=self.CLIENT)
        for i in range(self.NUM_DRONES):
            for r in range(self.LIDAR_RAYS):
                hit = results[i * self.LIDAR_RAYS + r]
                dist = hit[2] * self.LIDAR_RANGE if hit[0] != -1 else self.LIDAR_RANGE
                self.lidar_cache[i, r] = dist / self.LIDAR_RANGE

    def _sample_targets(self):
        self.target_positions = np.zeros_like(self.target_positions)
        bounds = 6.0
        for t in range(self.NUM_TARGETS):
            x = np.random.uniform(-bounds, bounds)
            y = np.random.uniform(-bounds, bounds)
            self.target_positions[t] = np.array([x, y, self.PREDATOR_ALT], dtype=np.float32)
        self.target_urgency[:] = 1.0
        if self.GUI:
            self._draw_targets()

    def _draw_targets(self):
        for marker in self._target_markers:
            p.removeUserDebugItem(marker, physicsClientId=self.CLIENT)
        self._target_markers.clear()
        for pos in self.target_positions:
            marker = p.addUserDebugText(
                "T",
                textPosition=pos + np.array([0.0, 0.0, 0.2]),
                textColorRGB=[1, 1, 0],
                textSize=1.2,
                lifeTime=0,
                physicsClientId=self.CLIENT,
            )
            self._target_markers.append(marker)

    def _spawn_predator(self):
        if self.predator_id is None:
            self.predator_id = p.loadURDF(
                "sphere2.urdf",
                [0, 0, self.PREDATOR_ALT],
                physicsClientId=self.CLIENT,
            )
        self._reset_predator_pose()

    def _reset_predator_pose(self):
        self.predator_pos = np.array([0.0, 0.0, self.PREDATOR_ALT], dtype=np.float32)
        self.predator_vel = np.zeros(3, dtype=np.float32)
        if self.predator_id is not None:
            p.resetBasePositionAndOrientation(
                self.predator_id,
                self.predator_pos,
                [0, 0, 0, 1],
                physicsClientId=self.CLIENT,
            )

    def _update_post_step_state(self):
        self._update_predator_agent()
        self._update_target_assignments()
        self._update_target_urgency()
        self._last_contacts = p.getContactPoints(physicsClientId=self.CLIENT)
        self.collision_flag = len(self._last_contacts) > 0

    def _update_predator_agent(self):
        if self.predator_id is None:
            return
        if self.NUM_DRONES == 0:
            return
        target_idx = np.argmin(np.linalg.norm(self.pos[:, :2], axis=1))
        direction = self.pos[target_idx] - self.predator_pos
        direction[2] = 0.0
        norm = np.linalg.norm(direction) + 1e-6
        velocity = (direction / norm) * self.PREDATOR_SPEED
        self.predator_vel = velocity
        self.predator_pos = self.predator_pos + velocity * self.CTRL_TIMESTEP
        self.predator_pos[2] = self.PREDATOR_ALT
        p.resetBasePositionAndOrientation(
            self.predator_id,
            self.predator_pos,
            [0, 0, 0, 1],
            physicsClientId=self.CLIENT,
        )

    def _update_target_assignments(self):
        for i in range(self.NUM_DRONES):
            distances = np.linalg.norm(self.target_positions[:, :2] - self.pos[i, :2], axis=1)
            objective = distances - self.target_urgency
            self.drone_target_assignment[i] = int(np.argmin(objective))

    def _update_target_urgency(self):
        for t in range(self.NUM_TARGETS):
            distances = np.linalg.norm(self.pos[:, :2] - self.target_positions[t, :2], axis=1)
            inside = distances < self.TARGET_RADIUS
            if np.any(inside):
                self.target_urgency[t] = np.clip(
                    self.target_urgency[t] - self.TARGET_DECAY * inside.sum(), 0.0, 1.0
                )

    def _formation_error(self) -> float:
        if self.NUM_DRONES < 2:
            return 0.0
        centroid = np.mean(self.pos[:, :2], axis=0)
        dists = np.linalg.norm(self.pos[:, :2] - centroid, axis=1)
        desired = max(1.0, 0.5 * np.sqrt(self.NUM_DRONES))
        return float(np.mean(np.abs(dists - desired)))

    def _navigation_cost(self) -> float:
        costs = []
        for i in range(self.NUM_DRONES):
            target = self.target_positions[self.drone_target_assignment[i]]
            costs.append(np.linalg.norm(self.pos[i, :2] - target[:2]))
        return float(np.mean(costs)) if costs else 0.0

    def _coverage_score(self) -> float:
        return float(np.sum(1.0 - self.target_urgency) / max(1, self.NUM_TARGETS))

    def _evasion_penalty(self) -> float:
        distances = np.linalg.norm(self.pos - self.predator_pos, axis=1)
        penalty = np.maximum(0.0, self.SAFE_RADIUS - distances)
        return float(np.mean(penalty ** 2))
