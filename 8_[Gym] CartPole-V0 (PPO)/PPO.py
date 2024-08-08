import gym
import torch
import random
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from gym.utils.env_checker import check_env
from gym.wrappers import TimeLimit 
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class PolicyNet(torch.nn.Module):
    ''' 策略网络是一个两层 MLP '''
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))             # (1, hidden_dim)
        x = F.softmax(self.fc2(x), dim=1)   # (1, output_dim)
        return x

class VNet(torch.nn.Module):
    ''' 价值网络是一个两层 MLP '''
    def __init__(self, input_dim, hidden_dim):
        super(VNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PPO(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_range, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):
        super().__init__()
        self.actor = PolicyNet(state_dim, hidden_dim, action_range).to(device)
        self.critic = VNet(state_dim, hidden_dim).to(device) 
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.device = device
        self.gamma = gamma
        self.lmbda = lmbda      # GAE 参数
        self.epochs = epochs    # 一条轨迹数据用来训练的轮数
        self.eps = eps          # PPO 中截断范围的参数
        self.device = device        

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        state = state.unsqueeze(0)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def compute_advantage(self, gamma, lmbda, td_delta):
        ''' 广义优势估计 GAE '''
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(np.array(advantage_list), dtype=torch.float)

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1-dones)
        td_delta = td_target - self.critic(states)
        advantage = self.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        # 用刚采集的一条轨迹数据训练 epochs 轮
        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))                   # PPO损失函数
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            
            # 更新网络参数
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

if __name__ == "__main__":
    def moving_average(a, window_size):
        ''' 生成序列 a 的滑动平均序列 '''
        cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
        middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
        r = np.arange(1, window_size-1, 2)
        begin = np.cumsum(a[:window_size-1])[::2] / r
        end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
        return np.concatenate((begin, middle, end))

    def set_seed(env, seed=42):
        ''' 设置随机种子 '''
        env.action_space.seed(seed)
        env.reset(seed=seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    state_dim = 4               # 环境观测维度
    action_range = 2            # 环境动作空间大小
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 200
    hidden_dim = 64
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # build environment
    env_name = 'CartPole-v0'
    env = gym.make(env_name, render_mode='rgb_array')
    check_env(env.unwrapped)    # 检查环境是否符合 gym 规范
    set_seed(env, 0)

    # build agent
    agent = PPO(state_dim, hidden_dim, action_range, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)

    '''
    # TensorBoard writer
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    writer = SummaryWriter(f"logs/PPO")
    #writer = SummaryWriter(f"logs/PPO/{TIMESTAMP}")
    '''

    # start training
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'next_actions': [],
                    'rewards': [],
                    'dones': []
                }
                state, _ = env.reset()

                # 以当前策略交互得到一条轨迹
                while True:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    next_action = agent.take_action(next_state)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['next_actions'].append(next_action)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(terminated or truncated)
                    state = next_state
                    episode_return += reward
                                        
                    if terminated or truncated:
                        break
                    #env.render()

                # 用当前策略收集的数据进行 on-policy 更新
                agent.update(transition_dict)

                # 更新进度条
                return_list.append(episode_return)
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % episode_return,
                    'ave return':
                    '%.3f' % np.mean(return_list[-10:])
                })
                pbar.update(1)

                '''
                writer.add_scalars(
                    main_tag='return', 
                    tag_scalar_dict={f'hidden{hidden_dim}':episode_return}, 
                    global_step=i*int(num_episodes / 10) + i_episode
                )
                '''

    '''
    writer.add_hparams(
        hparam_dict={
            'actor_lr': actor_lr, 
            'critic_lr': critic_lr,
            'hidden_dim': hidden_dim,
            'gamma': gamma,
            'lmbda': lmbda,
            'eps': eps,
            'num_episodes': num_episodes
        },
        metric_dict={
            'hparam/ave return': np.mean(return_list), 
        }
    )
    writer.close()
    '''
    
    # show policy performence
    mv_return_list = moving_average(return_list, 29)
    episodes_list = list(range(len(return_list)))
    plt.figure(figsize=(12,8))
    plt.plot(episodes_list, return_list, label='raw', alpha=0.5)
    plt.plot(episodes_list, mv_return_list, label='moving ave')
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'{agent._get_name()} on CartPole-V0')
    plt.legend()
    plt.savefig(f'./result/{agent._get_name()}.png')
    plt.show()                           
    