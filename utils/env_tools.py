"""Tools for HARL."""
import os
import random
import numpy as np
import torch
from envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv


def check(value):
    """Check if value is a numpy array, if so, convert it to a torch tensor."""
    output = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    return output


def get_shape_from_obs_space(obs_space):
    """Get shape from observation space.
    Args:
        obs_space: (gym.spaces or list) observation space
    Returns:
        obs_shape: (tuple) observation shape
    """
    if obs_space.__class__.__name__ == "Box":
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == "list":
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape


def get_shape_from_act_space(act_space):
    """Get shape from action space.
    Args:
        act_space: (gym.spaces) action space
    Returns:
        act_shape: (tuple) action shape
    """
    if act_space.__class__.__name__ == "Discrete":
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    return act_shape

def make_env(args):
    from envs.mpe_domain.environment import MultiAgentEnv
    import envs.mpe_domain.scenarios as scenarios
    # load scenario
    scenario = scenarios.load(args['scenario_name'] + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    args['n_players'] = env.n # 包含敌人的所有玩家个数
    args['n_agents'] = env.n - args['num_adversaries'] # 需要控制的玩家个数，虽然敌人也可以控制吗，但是双方都学习的话需要不同的算法
    args['obs_shape'] = [env.observation_space[i].shape[0] for i in range(args['n_agents'])] # 每一维代表该agent的obs维度
    action_shape = []
    for content in env.action_space:
        action_shape.append(content.n)
    args['action_shape'] = action_shape[:args['n_agents']] # 每一维代表该agent的act维度
    args['high_action'] = 1
    args['low_action'] = -1
    return env, args

def make_train_env(env_name, seed, n_threads, env_args):
    """Make env for training."""
    def get_env_fn(rank):
        def init_env():
            if "mujoco" in env_name:
                from envs.mujoco.multitask import MultitaskMujoco
                env = MultitaskMujoco(env_args)
                env.env.seed(seed)
                return env
            elif "dexhands" in env_name:
                from envs.dexhands.env_utils import make_single_env
                env = make_single_env(env_args, rank)
                return env
            elif "mpe" in env_name:
                from envs.mpe_domain.make_env import make_env
                env = make_env(env_args['scenario_name'])
                return env
            else:
                print("Can not support the " + env_name + "environment.")
                raise NotImplementedError
        return init_env

    if "dexhands" in env_name:
        from envs.dexhands.multitask import MultitaskDexHands
        return MultitaskDexHands(env_args, [get_env_fn(i) for i in range(env_args["n_tasks"])])
    else:
        if n_threads == 1:
            return ShareDummyVecEnv([get_env_fn(0)])
        else:
            return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_eval_env(env_name, seed, n_threads, env_args):
    """Make env for evaluation."""
    def get_env_fn(rank):
        def init_env():
            if "mujoco" in env_name:
                from envs.mujoco.multitask import MultitaskMujoco
                env = MultitaskMujoco(env_args)
                env.env.seed(seed)
                return env
            else:
                print("Can not support the " + env_name + "environment.")
                raise NotImplementedError
        return init_env

    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def set_seed(args):
    """Seed the program."""
    if not args["seed_specify"]:
        args["seed"] = np.random.randint(1000, 10000)
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    os.environ["PYTHONHASHSEED"] = str(args["seed"])
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])
    torch.backends.cudnn.deterministic = args['torch_deterministic']


def get_num_agents(env, env_args, envs):
    """Get the number of agents in the environment."""
    if env == "mujoco":
        return envs.n_agents
    elif env == "dexhands":
        return 2