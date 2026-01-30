from enum import Enum
class DroneModel(Enum):
    """Drone models enumeration class."""
    CF2X = "cf2x"  # Bitcraze Craziflie 2.0 in the X configuration
    CF2P = "cf2p"  # Bitcraze Craziflie 2.0 in the + configuration
    RACE = "racer"  # Racer drone in the X configuration

################################################################################

class Physics(Enum):
    """Physics implementations enumeration class."""
    PYB = "pyb"  # Base PyBullet physics update
    DYN = "dyn"  # Explicit dynamics model
    PYB_GND = "pyb_gnd"  # PyBullet physics update with ground effect
    PYB_DRAG = "pyb_drag"  # PyBullet physics update with drag
    PYB_DW = "pyb_dw"  # PyBullet physics update with downwash
    PYB_GND_DRAG_DW = "pyb_gnd_drag_dw"  # PyBullet physics update with ground effect, drag, and downwash

################################################################################

class ImageType(Enum):
    """Camera capture image type enumeration class."""
    RGB = 0  # Red, green, blue (and alpha)
    DEP = 1  # Depth
    SEG = 2  # Segmentation by object id
    BW = 3  # Black and white

################################################################################

class ActionType(Enum):
    RPM = "rpm"  # RPMS  # 基于电机转速的控制方式
    PID = "pid"  # PID control #每个电动机的转速是根据 PID 算法计算得出的，这个算法会根据无人机的当前状态（如角度、速度等）来调整每个电动机的 RPM，使得无人机保持所需姿态或轨迹
    VEL = "vel"  # Velocity input (using PID control) # 控制系统会根据目标速度（例如 x、y、z 轴的速度或角速度）来计算每个电动机的转速。与 PID 类似，但这里的目标是速度，而不是位置或姿态。
    ONE_D_RPM = "one_d_rpm"  # 1D (identical input to all motors) with RPMs # 所有电动机的转速相同，通常控制整个平台的悬浮高度或直线加速等简单任务。
    ONE_D_PID = "one_d_pid"  # 1D (identical input to all motors) with PID control 3 与 ONE_D_RPM 不同，虽然所有电动机的输入是相同的，但这里的控制方式会根据 PID 算法来调节转速。这样可以在保持简单的情况下，对飞行器的控制进行更加精细的调节

################################################################################

class ObservationType(Enum):
    """Observation type enumeration class."""
    KIN = "kin"  # Kinematic information (pose, linear and angular velocities)
    RGB = "rgb"  # RGB camera capture in each drone's POV