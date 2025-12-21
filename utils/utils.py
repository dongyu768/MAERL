import yaml
import torch
import numpy as np
from torch.autograd import Variable
import random, pickle, copy, argparse
import sys, os

def load_config(config_path='algo_models/config.yaml'):
    '''加载参数文件'''
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def hard_update(target, source):
    """Hard update (clone) from target network to source
    """

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

    #Signature transfer if applicable
    try:
        target.wwid[0] = source.wwid[0]
    except:
	    None

def list_mean(l):
    """compute avergae from a list
    """
    if len(l) == 0: return None
    else: return sum(l)/len(l)

def compute_stats(tensor, tracker):
    """Computes stats from intermediate tensors
     """
    tracker['min'] = torch.min(tensor).item()
    tracker['max'] = torch.max(tensor).item()
    tracker['mean'] = torch.mean(tensor).item()
    tracker['std'] = torch.std(tensor).item()

def to_tensor(ndarray, volatile=False, requires_grad=False):
    """numpy --> Variable

    Parameters:
        ndarray (ndarray): ndarray
        volatile (bool): create a volatile tensor?
        requires_grad (bool): tensor requires gradients?

    Returns:
        var (variable): variable
    """

    if isinstance(ndarray, list): ndarray = np.array(ndarray)
    return Variable(torch.from_numpy(ndarray).float(), volatile=volatile, requires_grad=requires_grad)

def to_numpy(var):
    """Tensor --> numpy

    Parameters:
        var (tensor): tensor

    Returns:
        var (ndarray): ndarray
    """
    return var.data.numpy()

def pprint(l):
    """Pretty print

    Parameters:
        l (list/float/None): object to print

    Returns:
        pretty print str
    """

    if isinstance(l, list):
        if len(l) == 0: return None
        else: ['%.2f'%item for item in l]
    elif isinstance(l, dict):
        return l
    elif isinstance(l, tuple):
        return ['%.2f'%item for item in l]
    else:
        if l == None: return None
        else: return '%.2f'%l

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Tracker(): #Tracker
    """Tracker class to log progress and save metrics periodically

    Parameters:
        save_folder (str): Folder name for saving progress
        vars_string (list): List of metric names to log
        project_string: (str): String decorator for metric filenames

    Returns:
        None
    """

    def __init__(self, save_folder, vars_string, project_string, save_iteration=1, conv_size=1):
        self.vars_string = vars_string; self.project_string = project_string
        self.foldername = save_folder
        self.all_tracker = [[[],0.0,[]] for _ in vars_string] #[Id of var tracked][fitnesses, avg_fitness, csv_fitnesses]
        self.counter = 0
        self.conv_size = conv_size
        self.save_iteration = save_iteration
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername)


    def update(self, updates, generation):
        """Add a metric observed

        Parameters:
            updates (list): List of new scoresfor each tracked metric
            generation (int): Current gen

        Returns:
            None
        """

        self.counter += 1
        for update, var in zip(updates, self.all_tracker):
            if update == None: continue
            var[0].append(update)

        #Constrain size of convolution
        for var in self.all_tracker:
            if len(var[0]) > self.conv_size: var[0].pop(0)

        #Update new average
        for var in self.all_tracker:
            if len(var[0]) == 0: continue
            var[1] = sum(var[0])/float(len(var[0]))

        if self.counter % self.save_iteration == 0:  # Save to csv file
            for i, var in enumerate(self.all_tracker):
                if len(var[0]) == 0: continue
                var[2].append(np.array([generation, var[1]]))
                filename = self.foldername + self.vars_string[i] + self.project_string
                np.savetxt(filename, np.array(var[2]), fmt='%.3f', delimiter=',')
