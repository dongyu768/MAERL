from envs.multiagent.environment import MultiAgentEnv
import envs.multiagent.scenarios as scenarios

def make_env(args):
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