import numpy as np
import threading
import random
import torch
from torch.multiprocessing import Manager

class Buffer:
    def __init__(self, args):
        self.size = args['buffer_size']
        self.args = args
        # memory management
        self.current_size = 0
        # create the buffer to store info
        self.buffer = dict()
        for i in range(self.args['n_agents']):
            self.buffer['o_%d' % i] = np.empty([self.size, self.args['obs_shape'][i]]) # (500000, )
            self.buffer['u_%d' % i] = np.empty([self.size, self.args['action_shape'][i]])
            self.buffer['r_%d' % i] = np.empty([self.size])
            self.buffer['o_next_%d' % i] = np.empty([self.size, self.args['obs_shape'][i]])
        # thread lock
        self.lock = threading.Lock()
    # store the episode
    def store_episode(self, o, u, r, o_next):
        idxs = self._get_storage_idx(inc=1) # 以transition的形式存，每次只存一条经验
        for i in range(self.args['n_agents']):
            with self.lock:
                self.buffer['o_%d' % i][idxs] = o[i]
                self.buffer['u_%d' % i][idxs] = u[i]
                self.buffer['r_%d' % i][idxs] = r[i]
                self.buffer['o_next_%d' % i][idxs] = o_next[i]
    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size-self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx
    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idx]
        return temp_buffer

class EvoBuffer:
	"""Cyclic Buffer stores experience tuples from the rollouts
		Parameters:
			capacity (int): Maximum number of experiences to hold in cyclic buffer
		"""

	def __init__(self, capacity, buffer_gpu):
		self.capacity = capacity; self.buffer_gpu = buffer_gpu
		self.manager = Manager()
		self.tuples = self.manager.list() #Temporary shared buffer to get experiences from processes
		self.s = []; self.ns = []; self.a = []; self.r = []; self.done = []; self.global_reward = []

		# Temporary tensors that cane be loaded in GPU for fast sampling during gradient updates (updated each gen) --> Faster sampling - no need to cycle experiences in and out of gpu 1000 times
		self.sT = None; self.nsT = None; self.aT = None; self.rT = None; self.doneT = None; self.global_rewardT = None

		self.pg_frames = 0; self.total_frames = 0


	def data_filter(self, exp):

		self.s.append(exp[0])
		self.ns.append(exp[1])
		self.a.append(exp[2])
		self.r.append(exp[3])
		self.done.append(exp[4])
		self.global_reward.append(exp[5])
		self.pg_frames += 1
		self.total_frames += 1


	def referesh(self):
		"""Housekeeping
			Parameters:
				None
			Returns:
				None
		"""

		# Add ALL EXPERIENCE COLLECTED TO MEMORY concurrently
		for _ in range(len(self.tuples)):
			exp = self.tuples.pop()
			self.data_filter(exp)

		#Trim to make the buffer size < capacity
		while self.__len__() > self.capacity:
			self.s.pop(0); self.ns.pop(0); self.a.pop(0); self.r.pop(0); self.done.pop(0); self.global_reward.pop(0)


	def __len__(self):
		return len(self.s)

	def sample(self, batch_size):
		"""Sample a batch of experiences from memory with uniform probability
			   Parameters:
				   batch_size (int): Size of the batch to sample
			   Returns:
				   Experience (tuple): A tuple of (state, next_state, action, shaped_reward, done) each as a numpy array with shape (batch_size, :)
		   """
		#Uniform sampling
		ind = random.sample(range(len(self.sT)), batch_size)

		return self.sT[ind], self.nsT[ind], self.aT[ind], self.rT[ind], self.doneT[ind], self.global_rewardT[ind]
		#return np.vstack([self.s[i] for i in ind]), np.vstack([self.ns[i] for i in ind]), np.vstack([self.a[i] for i in ind]), np.vstack([self.r[i] for i in ind]), np.vstack([self.done[i] for i in ind])


	def tensorify(self):
		"""Method to save experiences to drive
			   Parameters:
				   None
			   Returns:
				   None
		   """
		self.referesh() #Referesh first

		if self.__len__() >1:

			self.sT = torch.tensor(np.vstack(self.s))
			self.nsT = torch.tensor(np.vstack(self.ns))
			self.aT = torch.tensor(np.vstack(self.a))
			self.rT = torch.tensor(np.vstack(self.r))
			self.doneT = torch.tensor(np.vstack(self.done))
			self.global_rewardT = torch.tensor(np.vstack(self.global_reward))
			if self.buffer_gpu:
				self.sT = self.sT.cuda()
				self.nsT = self.nsT.cuda()
				self.aT = self.aT.cuda()
				self.rT = self.rT.cuda()
				self.doneT = self.doneT.cuda()
				self.global_rewardT = self.global_rewardT.cuda()





