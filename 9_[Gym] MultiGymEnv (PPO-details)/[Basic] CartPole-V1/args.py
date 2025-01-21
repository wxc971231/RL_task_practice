import argparse
import os
from distutils.util import strtobool

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
    parser.add_argument(
        "--capture-video", 
        type=lambda x: bool(strtobool(x)), 
        default=False, 
        nargs="?", 
        const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)"
    )
    parser.add_argument(
        "--video-trigger", 
        type=int, 
        default=0, 
        help="If set video_trigger > 0, the episode trigger for RecordVideo will be `lambda x: x % video_trigger==0`. Otherwise the default episode trigger will be used" 
    )
    
    # Algorithm specific arguments
    parser.add_argument(
        "--num-envs", 
        type=int, 
        default=4,
        help="the number of parallel game environments"
    )
    parser.add_argument(
        "--num-steps", 
        type=int, 
        default=128,
        help="the number of steps to run in each environment per policy rollout"
    )   # 每次 policy rollout 收集数据量（step为单位）为 batch_size = num_steps * num_envs，用这些数据来更新策略
    parser.add_argument(
        "--anneal-lr", 
        type=lambda x: bool(strtobool(x)), 
        default=True, 
        nargs="?", 
        const=True,
        help="Toggle learning rate annealing for policy and value networks"
    )   
    parser.add_argument(
        "--gae", 
        type=lambda x: bool(strtobool(x)), 
        default=True, 
        nargs="?", 
        const=True,
        help="Use GAE for advantage computation"
    )
    parser.add_argument(
        "--gamma", 
        type=float, 
        default=0.99,
        help="the discount factor gamma"
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="the lambda for the general advantage estimation"
    )
    parser.add_argument(
        "--num-minibatches", 
        type=int, 
        default=4,
        help="the number of mini-batches"
    )
    parser.add_argument(
        "--update-epochs", 
        type=int, 
        default=4,
        help="the K epochs to update the policy"
    )
    parser.add_argument(
        "--norm-adv", 
        type=lambda x: bool(strtobool(x)), 
        default=True, 
        nargs="?", 
        const=True,
        help="Toggles advantages normalization"
    )
    parser.add_argument(
        "--clip-coef", 
        type=float, 
        default=0.2,
        help="the surrogate clipping coefficient"
    )
    parser.add_argument(
        "--clip-vloss", 
        type=lambda x: bool(strtobool(x)), 
        default=True, nargs="?", 
        const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper."
    )
    parser.add_argument(
        "--ent-coef", 
        type=float, 
        default=0.01,
        help="coefficient of the entropy"
    )
    parser.add_argument(
        "--vf-coef", 
        type=float, 
        default=0.5,
        help="coefficient of the value function"
    )
    parser.add_argument(
        "--max-grad-norm", 
        type=float, 
        default=0.5,
        help="the maximum norm for the gradient clipping"
    )
    parser.add_argument(
        "--target-kl", 
        type=float, 
        default=None,   # 0.015 in OpenAI spinning up
        help="the target KL divergence threshold"
    )

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args