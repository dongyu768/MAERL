import yaml
import torch
import numpy as np
from torch.autograd import Variable

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
