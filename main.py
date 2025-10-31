import os, sys
import torch
import time
import datetime
import random
import numpy as np
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)
sys.path.append(project_root)
from utils import *
from envs.utils import make_env
from runner import Runner


def main():
    # load config
    config = load_config()
    args = config['config']

    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.backends.cudnn.deterministic = args['torch_deterministic']

    result_path = f"{args['base_path']}/{args['scenario_name']}/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    log_dir = f"{result_path}/summary"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    args['log_dir'] = log_dir
    save_dir = f"{result_path}/models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args['save_dir'] = save_dir
    

    # from env wrapper make env
    env, args = make_env(args)
    runner = Runner(args, env)
    # evaluate and train
    start_time = time.time()
    if args.get('evaluate', False):
        runner.evaluate()
    else:
        runner.run()
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"training duration: {training_time}")

if __name__ == "__main__":
    main()