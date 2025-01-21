import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(base_path)

import random
import time
import gym
import numpy as np
import functools
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.distributions.categorical import Categorical
from args import parse_args
from torch.utils.tensorboard import SummaryWriter

def make_env(gym_id, seed=1, idx=0, capture_video=False, run_name='', epi_trigger=0):
    env = gym.make(gym_id, render_mode='rgb_array')
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if capture_video:
        if idx == 0:
            if epi_trigger > 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda x: x%epi_trigger==0)
            else:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

class Agent(nn.Module):
    def __init__(self, envs):
        def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
            '''PPO implementation detail 2'''
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)
            return layer

        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            _layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, 1), std=1.),
        )
        self.actor = nn.Sequential(
            _layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),    # 用更小的标准差初始化 actor 以保证各个动作的初始选择概率差不多
        )
    
    def get_value(self, x):
        '''
        x: An observation for vector envs, shape=(num_envs, obs_dim) 
        '''
        return self.critic(x)               # (env_nums, 1)


    def get_action_and_value(self, x, action=None):
        '''
        x: An observation for vector envs, shape=(num_envs, obs_dim) 
        '''
        logits = self.actor(x)              # (env_nums, act_value_range)
        probs = Categorical(logits=logits)  # 这就是softmax计算出的概率
        if action is None:
            action = probs.sample()         # (env_nums, act_dim), if act_dim==1 then (env_nums, )
        log_prob = probs.log_prob(action)   # (env_nums, )
        entropy = probs.entropy()           # (env_nums, )
        value = self.critic(x)              # (env_nums, 1)
        return action, log_prob, entropy, value

def get_args_ready():
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
            dir = Path(f'{base_path}/Wandb'),
            sync_tensorboard=True,
            config=vars(args),
            name=args.run_name,
            monitor_gym=True,       # 上传 gym 环境中录制的视频
            save_code=True,         # 上传代码副本
        )

    # Setup TensorBoard
    writer = SummaryWriter(f"runs/{args.run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Setup Vectorized environments (PPO implementation detail 1)
    envs = [functools.partial(make_env, args.gym_id, args.seed+i, i, args.capture_video, args.run_name, args.video_trigger) \
        for i in range(args.num_envs)]    # 使用 functools.partial 冻结每次循环传给 make_env 的参数 
    envs = gym.vector.SyncVectorEnv(envs)  
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), 'only discrete action space is supported'

    # build PPO Agent
    agent = Agent(envs).to(device)
    
    # build optimizer
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5) # PPO implementation detail 3: use eps=1e-5

    # ALGO Logic: Storage setup (这些变量存储一次 policy rollout 收集到的数据，step 总量为 batch_size = num_steps * num_envs)
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device) # (step_nums, env_nums, obs_dim)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)  # (step_nums, env_nums, act_dim)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)                                  # (step_nums, env_nums)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)                                   # (step_nums, env_nums)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)                                     # (step_nums, env_nums)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)                                    # (step_nums, env_nums)                                   

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)        # (env_nums, obs_dim)
    next_done = torch.zeros(args.num_envs).to(device)   # (env_nums, )
    num_updates = args.total_timesteps // args.batch_size

    # training loop
    for update in range(1, 1 + num_updates):
        if args.anneal_lr:
            # PPO implementation detail 4: linearly decrease learning rate
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # policy rollout 得到一个 batch 数据
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO Logic: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()  # (env_nums, )
            logprobs[step] = logprob            # (env_nums, )
            actions[step] = action              # (env_nums, act_dim), if act_dim==1 then (env_nums, )
            
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            next_obs = torch.Tensor(next_obs).to(device)                # (env_nums, obs_dim)
            done = np.any(np.vstack((terminated, truncated)), axis=0)   
            next_done = torch.Tensor(done).to(device)                   # (env_nums, )
            rewards[step] = torch.tensor(reward).to(device).view(-1)    # (env_nums, )
            
            if 'final_info' in info.keys():
                for item in info['final_info']:
                    if item is not None and "episode" in item.keys():
                        print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                        break

        # calculate advantage
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)   # bootstrap value if not done
            if args.gae:                                            
                # PPO implementation detail 5: Use GAE
                advantages = torch.zeros_like(rewards).to(device)   # (step_nums, env_nums)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values                       # reconstruct return with GAE
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values
    
        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)        # (batch_size, obs_dim), if obs_dim==1 then (batch_size, )
        b_logprobs = logprobs.reshape(-1)                                       # (batch_size, )
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)     # (batch_size, act_dim), if act_dim==1 then (batch_size, )
        b_advantages = advantages.reshape(-1)                                   # (batch_size, )
        b_returns = returns.reshape(-1)                                         # (batch_size, )
        b_values = values.reshape(-1)                                           # (batch_size, )

        # Optimizing the policy and value network
        clipfracs = []
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            # PPO implementation detail 6: update with minibacth instead of entire batch
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mb_obs = b_obs[mb_inds]                         # (minibatch_size, obs_dim), if obs_dim==1 then (batch_size, )
                mb_actions = b_actions[mb_inds]                 # (minibatch_size, act_dim), if act_dim==1 then (batch_size, )
                mb_logprobs = b_logprobs[mb_inds]               # (minibatch_size, )
                mb_advantages = b_advantages[mb_inds]           # (minibatch_size, )
                mb_return = b_returns[mb_inds]                  # (minibatch_size, )
                mb_values = b_values[mb_inds]                   # (minibatch_size, )
                
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions.long())
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()                          # (minibatch_size, )

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean() # 估计新旧策略间的 KL 散度，指示策略更新的程度
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()] # 指示 pg_loss clip 的频率

                if args.norm_adv:
                    # PPO implementation detail 7: Advantage normalization
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss (PPO implementation detail 8: Clipped objective)
                pg_loss1 = -mb_advantages * ratio               # (minibatch_size, )
                pg_loss2 = -mb_advantages * torch.clamp(        # (minibatch_size, )
                    ratio, 
                    1 - args.clip_coef, 
                    1 + args.clip_coef
                )                                          
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()  # 这个 loss 构造时用到了 -min(A,B) = max(-A,-B)

                # Value loss (PPO implementation detail 9: clipped value loss)
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - mb_return) ** 2
                    v_clipped = mb_values + torch.clamp(
                        newvalue - mb_values,
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - mb_return) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_return) ** 2).mean()
                
                # PPO implementation detail 10: Entropy regularization
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                # PPO implementation detail 11: Gloabal gradient clipping                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            
            # PPO implementation detail 12: Early stop if KL divergence has grown too large    
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break
        
        # explained_var 指示了 cirtic 的预测性能，越高越好
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

         # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    
    envs.close()
    writer.close()
    if args.track:
        wandb.finish()