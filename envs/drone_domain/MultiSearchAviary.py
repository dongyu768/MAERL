import numpy as np
from gymnasium import spaces
from envs.drone_domain.BaseRLAviary import BaseRLAviary
from envs.drone_domain.drone_utils.enums import DroneModel, Physics, ActionType, ObservationType


class MultiSearchAviary(BaseRLAviary):
    """Multi-agent RL problem: cooperative ground-target searching in a rectangle."""

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 2,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM,
                 # Search task params
                 num_targets: int = 8,
                 area_bounds=(( -2.0, 2.0), (-2.0, 2.0)),  # ((x_min,x_max),(y_min,y_max))
                 z_limit: float = 2.0,
                 detection_radius: float = 0.30
                 ):
        # episode length similar style with hover, but longer for searching
        self.EPISODE_LEN_SEC = 30

        # search task config
        self.NUM_TARGETS = int(num_targets)
        self.DETECTION_RADIUS = float(detection_radius)
        self.X_MIN, self.X_MAX = float(area_bounds[0][0]), float(area_bounds[0][1])
        self.Y_MIN, self.Y_MAX = float(area_bounds[1][0]), float(area_bounds[1][1])
        self.Z_MIN, self.Z_MAX = 0.0, float(z_limit)
        self.DOMAIN_DIAG = np.linalg.norm([self.X_MAX - self.X_MIN, self.Y_MAX - self.Y_MIN])

        # reward weights
        self.R_NEW_FOUND = 1.0
        self.R_STEP = 0.01
        self.DIST_PENALTY = 0.5  # scales mean min-distance to unfound targets

        # episode target states (set/reset at t=0 inside _computeObs)
        self.targets_xy = None
        self.targets_found = None

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

    # ----------------------- Internal helpers -----------------------

    def _reset_targets(self):
        # uniformly sample targets on ground within area bounds
        xs = np.random.uniform(self.X_MIN, self.X_MAX, size=(self.NUM_TARGETS, 1))
        ys = np.random.uniform(self.Y_MIN, self.Y_MAX, size=(self.NUM_TARGETS, 1))
        self.targets_xy = np.hstack([xs, ys]).astype(np.float32)
        self.targets_found = np.zeros(self.NUM_TARGETS, dtype=bool)

    def _targets_relative_features(self, drone_index: int):
        # per-target features wrt drone: [dx, dy, found] for each target
        # dx,dy are target - drone position on XY plane
        drone_xy = self.pos[drone_index, 0:2]
        dxdy = (self.targets_xy - drone_xy).astype(np.float32)  # shape (T,2)
        found = self.targets_found.astype(np.float32).reshape(-1, 1)  # shape (T,1)
        per_target = np.hstack([dxdy, found])  # (T,3)
        return per_target.reshape(-1)  # flatten to length 3*T

    def _nearest_unfound_distances(self):
        # returns list of min distance to any unfound target for each drone
        if self.targets_found is None or np.all(self.targets_found):
            return [0.0 for _ in range(self.NUM_DRONES)]
        remaining = self.targets_xy[~self.targets_found]  # (K,2)
        dists = []
        for i in range(self.NUM_DRONES):
            drone_xy = self.pos[i, 0:2]
            dd = np.linalg.norm(remaining - drone_xy, axis=1)
            dists.append(float(np.min(dd)))
        return dists

    # ----------------------- Spaces and Observations -----------------------

    def _observationSpace(self):
        """KIN observation + per-target features + action buffer."""
        if self.OBS_TYPE == ObservationType.RGB:
            # keep BaseRLAviary behavior for RGB
            return spaces.Box(low=0,
                              high=255,
                              shape=(self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4),
                              dtype=np.uint8)

        # KIN
        # base 12 dims per drone as in BaseRLAviary: [x,y,z, q1,q2,q3,q4, roll,pitch,yaw, vx,vy,vz, wx,wy,wz] there it picks 12: pos(3), rpy(3), vel(3), ang_v(3)
        base_lo = -np.inf
        base_hi = np.inf
        obs_lower_bound = np.array([[base_lo, base_lo, 0.0,   base_lo, base_lo, base_lo,   base_lo, base_lo, base_lo,   base_lo, base_lo, base_lo]
                                    for _ in range(self.NUM_DRONES)], dtype=np.float32)
        obs_upper_bound = np.array([[base_hi, base_hi, self.Z_MAX,  base_hi, base_hi, base_hi,  base_hi, base_hi, base_hi,  base_hi, base_hi, base_hi]
                                    for _ in range(self.NUM_DRONES)], dtype=np.float32)

        # add per-target features: for each target -> [dx, dy, found]
        # dx,dy bounded by domain size; found in [0,1]
        dxdy_bound = max(self.X_MAX - self.X_MIN, self.Y_MAX - self.Y_MIN)
        tgt_lo_row = np.array([-dxdy_bound, -dxdy_bound, 0.0] * self.NUM_TARGETS, dtype=np.float32)
        tgt_hi_row = np.array([+dxdy_bound, +dxdy_bound, 1.0] * self.NUM_TARGETS, dtype=np.float32)
        tgt_lo = np.vstack([tgt_lo_row for _ in range(self.NUM_DRONES)])
        tgt_hi = np.vstack([tgt_hi_row for _ in range(self.NUM_DRONES)])
        obs_lower_bound = np.hstack([obs_lower_bound, tgt_lo])
        obs_upper_bound = np.hstack([obs_upper_bound, tgt_hi])

        # add action buffer exactly like BaseRLAviary but here to match our new obs dims
        if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
            a_size = 4
        elif self.ACT_TYPE == ActionType.PID:
            a_size = 3
        elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
            a_size = 1
        else:
            raise RuntimeError("[ERROR] in MultiSearchAviary._observationSpace(): unknown action type")

        for _ in range(self.ACTION_BUFFER_SIZE):
            act_lo = np.array([[-1.0] * a_size for _ in range(self.NUM_DRONES)], dtype=np.float32)
            act_hi = np.array([[+1.0] * a_size for _ in range(self.NUM_DRONES)], dtype=np.float32)
            obs_lower_bound = np.hstack([obs_lower_bound, act_lo])
            obs_upper_bound = np.hstack([obs_upper_bound, act_hi])

        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    def _computeObs(self):
        """KIN: 12 + (3 * NUM_TARGETS) + action buffer; RGB: defer to base."""
        if self.OBS_TYPE == ObservationType.RGB:
            # defer to BaseRLAviary's implementation
            return super()._computeObs()

        # sample new targets at the beginning of each episode (step_counter == 0)
        if self.step_counter == 0 or (self.targets_xy is None):
            self._reset_targets()

        # base 12 dims per-drone
        obs_core = np.zeros((self.NUM_DRONES, 12), dtype=np.float32)
        for i in range(self.NUM_DRONES):
            st = self._getDroneStateVector(i)
            obs_core[i, :] = np.hstack([st[0:3], st[7:10], st[10:13], st[13:16]]).reshape(12,)

        # append per-target features for each drone
        rows = []
        for i in range(self.NUM_DRONES):
            tgt_feat = self._targets_relative_features(i)  # length 3*T
            row = np.hstack([obs_core[i, :], tgt_feat])
            rows.append(row.astype(np.float32))
        ret = np.vstack(rows).astype(np.float32)

        # append action buffer, robust to list-of-arrays using np.stack
        for i in range(self.ACTION_BUFFER_SIZE):
            buf_i = self.action_buffer[i]
            if isinstance(buf_i, np.ndarray):
                buf_arr = buf_i.astype(np.float32)
            else:
                # list of per-drone actions -> stack safely
                buf_arr = np.stack([np.array(buf_i[j], dtype=np.float32) for j in range(self.NUM_DRONES)], axis=0)
            ret = np.hstack([ret, buf_arr])
        return ret

    # ----------------------- Task logic -----------------------

    def _computeReward(self):
        """Team reward:
        +R_NEW_FOUND for each newly found target,
        -R_STEP every step,
        -DIST_PENALTY * mean(min_dist_to_unfound)/domain_diag as shaping until all found.
        """
        # 1) mark new discoveries
        new_hits = 0
        for t in range(self.NUM_TARGETS):
            if self.targets_found[t]:
                continue
            t_xy = self.targets_xy[t]
            # nearest drone distance on XY plane
            dmins = [np.linalg.norm(self.pos[i, 0:2] - t_xy) for i in range(self.NUM_DRONES)]
            if np.min(dmins) <= self.DETECTION_RADIUS:
                self.targets_found[t] = True
                new_hits += 1

        # 2) step cost
        reward = -self.R_STEP

        # 3) discovery bonus
        if new_hits > 0:
            reward += self.R_NEW_FOUND * float(new_hits)

        # 4) shaping towards remaining targets
        if not np.all(self.targets_found):
            dists = self._nearest_unfound_distances()  # per-drone min distance
            if len(dists) > 0 and self.DOMAIN_DIAG > 1e-6:
                reward += -self.DIST_PENALTY * (float(np.mean(dists)) / self.DOMAIN_DIAG)

        return float(reward)

    def _computeTerminated(self):
        """Episode terminates when all targets are found."""
        if self.targets_found is None:
            return False
        return bool(np.all(self.targets_found))

    def _computeTruncated(self):
        """Truncate on out-of-bounds, excessive tilt, or time limit."""
        # spatial/attitude constraints
        for i in range(self.NUM_DRONES):
            st = self._getDroneStateVector(i)
            x, y, z = st[0], st[1], st[2]
            roll, pitch = st[7], st[8]
            if (x < self.X_MIN or x > self.X_MAX or
                y < self.Y_MIN or y > self.Y_MAX or
                z < self.Z_MIN or z > self.Z_MAX):
                return True
            if (abs(roll) > .4 or abs(pitch) > .4):
                return True

        # time limit
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeInfo(self):
        """Info dict with progress stats."""
        found_cnt = int(np.sum(self.targets_found)) if self.targets_found is not None else 0
        return {
            "found": found_cnt,
            "remaining": int(self.NUM_TARGETS - found_cnt),
            "num_targets": int(self.NUM_TARGETS),
            "detection_radius": float(self.DETECTION_RADIUS)
        }
