import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(base_path)

import argparse
import random
import time
import numpy as np
import torch
from pathlib import Path
from distutils.util import strtobool
from torch.utils.tensorboard import SummaryWriter   # tensorboard --logdir runs

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name", 
        type=str, 
        default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment"
    )
    parser.add_argument(
        "--gym-id", 
        type=str, 
        default="CartPole-v1",
        help="the id of the gym environment"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=2.5e-4,
        help="the learning rate of the optimizer"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="seed of the experiment"
    )
    parser.add_argument(
        "--total-timesteps", 
        type=int, 
        default=25000,
        help="total timesteps of the experiments"
    )
    parser.add_argument(
        "--torch-deterministic", 
        type=lambda x: bool(strtobool(x)), 
        default=True,   # 若命令行中没有此参数，产生 default 值
        nargs="?",      # 这表示此参数在命令行中要出现0次或1次，即可有可无
        const=True,     # nargs="?" 时，若命令行中用 '--' toggled 了此参数但未赋值，产生 const 值
        help="Set is as True will help you to reproduce the result`"
    )
    parser.add_argument(
        "--cuda", 
        type=lambda x: bool(strtobool(x)), 
        default=True, 
        nargs="?", 
        const=True,
        help="if toggled, cuda will be enabled by default"
    )
    parser.add_argument(
        "--track", 
        type=lambda x: bool(strtobool(x)), 
        default=False, 
        nargs="?", 
        const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases"
    )
    parser.add_argument(
        "--wandb-project-name", 
        type=str, 
        default="ppo-basic",
        help="the wandb's project name"
    )
    parser.add_argument(
        "--wandb-entity", 
        type=str, 
        default=None,
        help="the entity (team) of wandb's project, by default (None) it will be the username"
    )

    args = parser.parse_args()
    return args

def get_args_ready():
    # 通过命令行传参比较繁琐，可在此修改实验参数
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() and args.cuda else 'cpu'

    args.exp_name = 'TEST'
    args.run_name = f"{args.gym_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
    args.track = True

    return args

if __name__ == '__main__':
    args = get_args_ready()
    device = torch.device(args.device)
    
    # Setup Wandb
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            dir=Path(f'{base_path}/Wandb'),
            sync_tensorboard=True,  # 同步 tensorboard log
            config=vars(args),      
            name=args.run_name,
            monitor_gym=True,       # 上传 gym 环境中录制的视频
            save_code=True,         # 上传代码副本
        )

    # Setup TensorBoard
    writer = SummaryWriter(f"{base_path}/runs/{args.run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # test data
    for i in range(100):
        writer.add_scalar("test_loss", i*2, global_step=i)

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    writer.close()
    if args.track:
        wandb.finish()