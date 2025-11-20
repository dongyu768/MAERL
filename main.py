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
from envs.env_wrapper import make_env
from runner import Runner


def main():
    # 获得全局配置参数
    config = load_config()
    args = config['config']
    # 设置随机种子
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.backends.cudnn.deterministic = args['torch_deterministic']
    # 根据训练场景和训练时间定义保存路径
    result_path = f"{args['base_path']}/{args['scenario_name']}-{datetime.datetime.now().strftime(f'%Y%m%d-%H%M')}"
    log_dir = f"{result_path}/summary"
    save_dir = f"{result_path}/models"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args['log_dir'] = log_dir
    args['save_dir'] = save_dir
    
    # 构建环境
    env, args = make_env(args)
    # 训练模型
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
    # TODO 增加保存当次训练参数模块
    # TODO 增加checkpoint断点保存模型功能，方便中断后继续训练

if __name__ == "__main__":
    main()