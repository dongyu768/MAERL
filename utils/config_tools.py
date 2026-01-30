"""Tools for loading and updating configs."""
import time
import os
import json
import yaml
from uu import Error
import datetime
from torch.utils.tensorboard import SummaryWriter

def get_defaults_yaml_args(algo, env):
    """Load config file for user-specified algo and env.
    Args:
        algo: (str) Algorithm name.
        env: (str) Environment name.
    Returns:
        algo_args: (dict) Algorithm config.
        env_args: (dict) Environment config.
    """
    base_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    base_cfg_path = os.path.join(base_path, "configs", "base.yaml")
    algo_cfg_path = os.path.join(base_path, "configs", "algo_configs", f"{algo}.yaml")
    env_cfg_path = os.path.join(base_path, "configs", "env_configs", f"{env}.yaml")

    with open(base_cfg_path, "r", encoding="utf-8") as file:
        base_args = yaml.load(file, Loader=yaml.FullLoader)
    with open(algo_cfg_path, "r", encoding="utf-8") as file:
        algo_args = yaml.load(file, Loader=yaml.FullLoader)
    with open(env_cfg_path, "r", encoding="utf-8") as file:
        env_args = yaml.load(file, Loader=yaml.FullLoader)
    return base_args, algo_args, env_args


def init_dir(args):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    result_path = os.path.join(args["base_path"], args["env_name"], args["algo_name"], f"{args['scenario_name']}-{timestamp}")
    log_dir = os.path.join(result_path, "summary")
    save_dir = os.path.join(result_path, "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    if args['save_vedio']:
        vedio_dir = os.path.join(result_path, "vedios")
        os.makedirs(vedio_dir, exist_ok=True)
        args['vedio_dir'] = vedio_dir
    args['log_dir'] = log_dir
    args['save_dir'] = save_dir
    return args

def init_writter(args):
    """Init summarywriter for logging results."""
    writter = SummaryWriter(args['log_dir'])
    return writter


def is_json_serializable(value):
    """Check if v is JSON serializable."""
    try:
        json.dumps(value)
        return True
    except Error:
        return False


def convert_json(obj):
    """Convert obj to a version which can be serialized with JSON."""
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) for k, v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj, "__name__") and not ("lambda" in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj, "__dict__") and obj.__dict__:
            obj_dict = {
                convert_json(k): convert_json(v) for k, v in obj.__dict__.items()
            }
            return {str(obj): obj_dict}

        return str(obj)


def save_config(args, algo_args, env_args, run_dir):
    """Save the configuration of the program."""
    config = {"main_args": args, "algo_args": algo_args, "env_args": env_args}
    config_json = convert_json(config)
    output = json.dumps(config_json, separators=(",", ":\t"), indent=4, sort_keys=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as out:
        out.write(output)