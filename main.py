import os, sys
import torch
import time
from datetime import timedelta
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
    args = load_config()

    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.backends.cudnn.deterministic = args['torch_deterministic']

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
    training_duration = str(timedelta(seconds=int(training_time)))
    print(f"training duration: {training_duration}")

if __name__ == "__main__":
    main()