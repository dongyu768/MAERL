import numpy as np
from envs.drone_domain.BaseRLAviary import BaseRLAviary
from envs.drone_domain.drone_utils.enums import DroneModel, Physics, ActionType, ObservationType

class MultiHoverAviary(BaseRLAviary):
    """Multi-agent RL problem: leader-follower."""
    def __init__(self,
                 drone_model:DroneModel=DroneModel.CF2X,
                 num_drones:int=2,
                 neighbourhood_radius:float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq:int=240,
                 ctrl_freq:int=30,
                 gui=False,
                 record=False,
                 obs:ObservationType=ObservationType.KIN,
                 act:ActionType=ActionType.RPM
                 ):
        self.EPISODE_LEN_SEC = 8
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
        self.TARGET_POS = self.INIT_XYZS + np.array([[0,0,1/(i+1)] for i in range(num_drones)])
        print("target position:", self.TARGET_POS)

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value."""
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        ret = 0
        for i in range(self.NUM_DRONES):
            ret += max(0, 2 - np.linalg.norm(self.TARGET_POS[i,:]-states[i][0:3])**4)
        return ret

    ################################################################################

    def _computeTerminated(self):
        """Computes the current done value."""
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        dist = 0
        terminated = []
        for i in range(self.NUM_DRONES):
            dist += np.linalg.norm(self.TARGET_POS[i,:]-states[i][0:3])
            if dist < .0001:
                return True
            else:
                return False

    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value."""
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        for i in range(self.NUM_DRONES):
            if (abs(states[i][0]) > 2.0 or abs(states[i][1]) > 2.0 or states[i][2] > 2.0  # Truncate when a drones is too far away
                    or abs(states[i][7]) > .4 or abs(states[i][8]) > .4  # Truncate when a drone is too tilted
            ):
                return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42}  #### Calculated by the Deep Thought supercomputer in 7.5M years

if __name__ == '__main__':
    DEFAULT_DRONES = DroneModel("cf2x")
    DEFAULT_PHYSICS = Physics("pyb")
    DEFAULT_GUI = True
    DEFAULT_RECORD = True
    DEFAULT_PLOT = True
    DEFAULT_USER_DEBUG_GUI = False
    DEFAULT_SIMULATION_FREQ_HZ = 500
    DEFAULT_CONTROL_FREQ_HZ = 25
    DEFAULT_OUTPUT_FOLDER = 'results'
    NUM_DRONES = 2
    INIT_XYZ = np.array([[.5 * i, .5 * i, .1] for i in range(NUM_DRONES)])
    INIT_RPY = np.array([[.0, .0, .0] for _ in range(NUM_DRONES)])
    DEFAULT_OBS = ObservationType("kin")
    DEFAULT_ACT = ActionType("rpm")
    test_env = MultiHoverAviary(drone_model=DEFAULT_DRONES,
                                num_drones=NUM_DRONES,
                                neighbourhood_radius = np.inf,
                                initial_xyzs=INIT_XYZ,
                                initial_rpys=INIT_RPY,
                                physics=DEFAULT_PHYSICS,
                                pyb_freq=DEFAULT_SIMULATION_FREQ_HZ,
                                ctrl_freq=DEFAULT_CONTROL_FREQ_HZ,
                                gui=DEFAULT_GUI,
                                record=DEFAULT_RECORD,
                                obs=DEFAULT_OBS,
                                act=DEFAULT_ACT
                                )
    print(test_env)

