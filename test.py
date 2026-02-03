import os, sys
import numpy as np
import argparse
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)
sys.path.append(project_root)
from utils.env_tools import make_env
from utils.config_tools import get_defaults_yaml_args, init_dir
from moviepy.editor import ImageSequenceClip
# import imageio

def main():
    # 配置参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo_name", type=str, default="maddpg",help="")
    parser.add_argument("--env_name", type=str, default="mpe",help="")   # [mpe, drone]
    parser.add_argument("--evaluate", type=bool, default=False ,help="")
    parser.add_argument("--eval_path", type=str, default="results/drone/maddpg/MultiSearchAviary-20260109-2338",help="")
    par_args = parser.parse_args()
    # 加载参数
    base_args, algo_args, env_args = get_defaults_yaml_args(par_args.algo_name, par_args.env_name)
    args = {}
    args.update(base_args);args.update(algo_args);args.update(env_args)
    args.update(vars(par_args))

    # 初始化目录
    if not args["evaluate"]:
        args = init_dir(args)
    env, args = make_env(args)
    env.reset()

    frames = []
    # writer = imageio.get_writer("record.mp4",fps=30,codec="libx264")
    for t in range(100):
        action_n = []
        for _ in range(args['n_players']):
                action_n.append(env.sample())
        obs_n, reward_n, done_n, info_n = env.step(action_n)
        print(f'========================{t}===========================')
        # print(f'action:\n{action_n}\nobs:\n{obs_n}\nreward:{reward_n}\ndone:{done_n}')
        frame = env.render()
        frames = frames.append(frame)
        # frame = frame[:, :, :3].astype(np.uint8)
        # h, w, _ = frame.shape
        # frame = frame[:h//2*2, :w//2*2]
        # writer.append_data(frame)

        # done_n 是 list[bool]，一般用 any/all 判断
        if any(done_n):
            break
    clip = ImageSequenceClip(frames, fps=30)
    clip.write_videofile(args['vedio_dir'] + "record.mp4", codec="libx264")

    env.close()
    # writer.close()


if __name__ == "__main__":
    main()