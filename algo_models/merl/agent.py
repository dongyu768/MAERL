from torch.multiprocessing import Manager
from algo_models.merl.neuroevolution import SSNE
from algo_models.merl.arch import MultiHeadActor
from common.replay_buffer import EvoBuffer

class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        ###Initalize neuroevolution module###
        self.evolver = SSNE(args)
        ########Initialize population
        self.manager = Manager()
        self.popn = self.manager.list()
        for _ in range(args['popn_size']):
            self.popn.append(MultiHeadActor(args))
            self.popn[-1].eval()
        
        #### INITIALIZE PG ALGO #####
        # algo_name = 'TD3' if args['is_matd3'] else 'DDPG'
        self.algo = SSNE(args, agent_id)

		#### Rollout Actor is a template used for MP #####
        self.rollout_actor = self.manager.list()
        self.rollout_actor.append(MultiHeadActor(args))

		#Initalize buffer
        self.buffer = [EvoBuffer(args) for _ in range(args['n_agents'])]

		#Agent metrics
        self.fitnesses = [[] for _ in range(args['popn_size'])]

		###Best Policy HOF####
        self.champ_ind = 0
    
    def update_parameters(self):
        td3args = {'policy_noise': 0.2, 'policy_noise_clip': 0.5, 'policy_ups_freq': 2, 'action_low': -1.0, 'action_high': 1.0}
        for agent_id, buffer in enumerate(self.buffer):
            if self.args['is_matd3'] or self.args['is_maddpg']: 
                buffer = self.buffer[0] #Hardcoded Hack for MADDPG

            buffer.referesh()
            if buffer.__len__() < 10 * self.args['batch_size']:
                buffer.pg_frames = 0
                return  ###BURN_IN_PERIOD
            buffer.tensorify()

            for _ in range(int(self.args['gradperstep'] * buffer.pg_frames)):
                s, ns, a, r, done, global_reward = buffer.sample(self.args['batch_size'])
                r*=self.args['reward_scaling']
                if self.args['use_gpu']:
                    s = s.cuda(); ns = ns.cuda(); a = a.cuda(); r = r.cuda(); done = done.cuda(); global_reward = global_reward.cuda()
                self.algo.update_parameters(s, ns, a, r, done, global_reward, agent_id, 1, **td3args)
            buffer.pg_frames = 0
    
    def evolve(self):
		## One gen of evolution ###
        if self.args['popn_size'] > 1: #If not no-evo

			#Net idices of nets that got evaluated this generation (meant for asynchronous evolution workloads)
            net_inds = [i for i in range(len(self.popn))] #Hack for a synchronous run

			#Evolve
            if self.args['rollout_size'] > 0: 
                self.champ_ind = self.evolver.evolve(self.popn, net_inds, self.fitnesses, [self.rollout_actor[0]])
            else: 
                self.champ_ind = self.evolver.evolve(self.popn, net_inds, self.fitnesses, [])

		#Reset fitness metrics
        self.fitnesses = [[] for _ in range(self.args['popn_size'])]

