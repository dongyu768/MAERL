"""Tools for HARL."""
import os
import random
import numpy as np
import torch
from envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv


def check(value):
    """Check if value is a numpy array, if so, convert it to a torch tensor."""
    output = torch.from_numpy(value).float().unsqueeze(0) if isinstance(value, np.ndarray) else value
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
        act_shape = act_space.n
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    return act_shape

def make_env(args):
    if args['env_name'] == 'mpe':
        from envs.mpe_domain.environment import MultiAgentEnv
        import envs.mpe_domain.scenarios as scenarios
        # load scenario
        scenario = scenarios.load(args['scenario_name'] + ".py").Scenario()
        # create world
        world = scenario.make_world(args['num_good'], args['num_adversary'], args['num_obstacle'], args['num_goal'])
        # create multiagent environment
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, done_callback=scenario.done)
        args['n_players'] = env.n # 包含敌人的所有玩家个数
        args['n_agents'] = env.n - args['num_adversary'] # 需要控制的玩家个数
        # args['obs_shape'] = [env.observation_space[i].shape[0] for i in range(args['n_agents'])] # 每一维代表该agent的obs维度
        args['obs_shape'] = [get_shape_from_obs_space(env.observation_space[i])[0] for i in range(args['n_players'])] # 每一维代表该agent的obs维度
        action_shape = []
        for content in env.action_space:
            action_shape.append(get_shape_from_act_space(content))
        args['action_shape'] = action_shape[:args['n_players']] # 每一维代表该agent的act维度
        args['high_action'] = 1
        args['low_action'] = -1
        # print(f"action_shape:{args['action_shape']}")
        # print(f"obs_shape:{args['obs_shape']}")
    elif args['env_name'] == 'drone':
        if args['scenario_name'] == 'MultiHoverAviary':
            from envs.drone_domain.MultiHoverAviary import MultiHoverAviary as scenv
        elif args['scenario_name'] == 'MultiSearchAviary':
            from envs.drone_domain.MultiSearchAviary import MultiSearchAviary as scenv
        elif args['scenario_name'] == 'MultiPEAviary':
            from envs.drone_domain.MultiPEAviary import MultiPEAviary as scenv
        # env = scenv(num_drones=args['n_agents'])
        env = scenv(num_drones=args['n_agents'], record=True)
        args['obs_shape'] = [get_shape_from_obs_space(env.observation_space)[1] for _ in range(args['n_agents'])] # 每一维代表该agent的obs维度
        args['action_shape'] = [get_shape_from_act_space(env.action_space)[1] for _ in range(args['n_agents'])]
        args['high_action'] = 1
        args['low_action'] = -1
    
    return env, args


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