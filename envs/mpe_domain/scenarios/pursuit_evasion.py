import numpy as np
from envs.mpe_domain.core import World, Agent, Landmark
from envs.mpe_domain.scenario import BaseScenario

class Scenario(BaseScenario):

    def make_world(self, num_good, num_adversaries, num_landmarks, num_goal):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = num_adversaries + num_good
        world.num_agents = num_agents
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True  # 智能体不能碰撞
            agent.silent = True
            agent.adversary = False if i < num_good else True
            agent.size = 0.05
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True   # 障碍物不可碰撞
            landmark.movable = False
            landmark.size = 0.02
        # add target
        world.targets = [Landmark() for i in range(num_goal)]
        for i, target in enumerate(world.targets):
            target.name = 'target %d' % i
            target.collide = False   # 目标点非实体，可碰撞
            target.movable = False
            target.size = 0.02
        # make initial conditions
        self.reset_world(world)
        return world
    
    def reset_world(self, world):
        # random properties for agents
        world.agents[0].color = np.array([0.85, 0.35, 0.35])
        for i in range(1, world.num_agents):
            world.agents[i].color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.15, 0.15])
        # random properties for targets
        for i, target in enumerate(world.targets):
            target.color = np.array([0.95, 0.95, 0.20])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, target in enumerate(world.targets):
            target.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            target.state.p_vel = np.zeros(world.dim_p)
    
    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        def sqdist(a_pos, b_pos):
            return np.sum(np.square(a_pos - b_pos))

        if agent.adversary:
            # adversary: 离任意一个target最近的平方距离
            return min([sqdist(agent.state.p_pos, t.state.p_pos) for t in world.targets])
        else:
            # good： 对每一个target，团队到该target的最近平方距离
            goods = self.good_agents(world)
            cover_sq_dists = []
            for t in world.targets:
                cover_sq_dists.append(min([sqdist(g.state.p_pos, t.state.p_pos) for g in goods]))
            return tuple(cover_sq_dists)
    
    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]
    
    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]
    
    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
    
    def agent_reward(self, agent, world):
        # Rewarded based on how close any good agent is to the goal landmark, and how far the adversary is from it
        shaped_reward = True
        shaped_adv_reward = True

        goods = self.good_agents(world)
        adversaries = self.adversaries(world)

        if shaped_reward:
            cover_cost = 0.0
            for t in world.targets:
                cover_cost += min([np.sqrt(np.sum(np.square(g.state.p_pos - t.state.p_pos))) for g in goods])
            pos_rew = -cover_cost
        else:
            # 所有target都被覆盖才给奖励
            eps = 2 * world.targets[0].size
            covered = True
            for t in world.targets:
                if min([np.sqrt(np.sum(np.square(g.state.p_pos - t.state.p_pos))) for g in goods]) > eps:
                    covered = False
                    break
            pos_rew = 5.0 if covered else 0.0
        
        if shaped_adv_reward:
            adv_rew = sum([min([np.sqrt(np.sum(np.square(a.state.p_pos - t.state.p_pos))) for t in world.targets]) for a in adversaries])
        else:
            eps = 2 * world.targets[0].size
            covered = True
            for t in world.targets:
                if min([np.sqrt(np.sum(np.square(a.state.p_pos - t.state.p_pos))) for a in adversaries]) > eps:
                    covered = False
                    break
            adv_rew = -5.0 if covered else 0.0
        
        return pos_rew + adv_rew
    
    def adversary_reward(self, agent, world):
        # Rewarded based on proximity to the goal landmark
        shaped_reward = True
        if shaped_reward:  # distance-based reward
            return -min([np.sum(np.square(agent.state.p_pos - t.state.p_pos)) for t in world.targets])
        else:  # proximity-based reward (binary)
            eps = 2 * world.targets[0].size
            dmin = min([np.sqrt(np.sum(np.square(agent.state.p_pos - t.state.p_pos))) for t in world.targets])
            return 5.0 if dmin < eps else 0.0
        
    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        target_pos = [t.state.p_pos - agent.state.p_pos for t in world.targets]

        # target_color = [t.color for t in world.targets]

        entity_pos = [l.state.p_pos - agent.state.p_pos for l in world.landmarks]

        # entity colors
        # entity_color = [l.color for l in world.landmarks]

        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        if not agent.adversary:
            return np.concatenate(target_pos + entity_pos + other_pos)
        else:
            return np.concatenate(entity_pos + other_pos)

    def done(self, agent, world):
        # 出界就结束（你也可以只对 good 判断）
        return np.any(np.abs(agent.state.p_pos) > 1.0)