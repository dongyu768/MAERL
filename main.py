import os, sys
import torch
import time
import datetime
import random
import numpy as np
import argparse
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)
sys.path.append(project_root)
from utils.env_tools import make_env, set_seed
from utils.config_tools import get_defaults_yaml_args, init_dir
from runners.runner import Runner
from runners.maerl_runner import MAERLRunner

def main():
    # 加载配置参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo_name", type=str, default="maerl",help="")
    parser.add_argument("--env_name", type=str, default="mpe",help="")
    parser.add_argument("--device", type=str, default="cuda:2",help="")
    par_args = parser.parse_args()
    
    base_args, algo_args, env_args = get_defaults_yaml_args(par_args.algo_name, par_args.env_name)
    args = {}
    args.update(base_args);args.update(algo_args);args.update(env_args)
    args.update(vars(par_args))

    # 设置随机种子
    set_seed(args)
    # 初始化目录
    args = init_dir(args)
    
    # 构建环境
    env, args = make_env(args)
    # 训练模型
    if args['algo_name'].lower() in ['maerl', 'merl']:
        runner = MAERLRunner(args, env)
    else:
        runner = Runner(args, env)
    runner = Runner(args, env)
    # 评估和训练
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