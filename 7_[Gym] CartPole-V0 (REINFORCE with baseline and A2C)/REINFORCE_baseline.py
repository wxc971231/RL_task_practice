import gym
import torch
import random
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from gym.utils.env_checker import check_env
from gym.wrappers import TimeLimit 

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

class REINFORCE_Baseline(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_range, lr_policy, lr_value, gamma, device):
        super().__init__()
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_range).to(device)
        self.v_net = VNet(state_dim, hidden_dim).to(device)
        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=lr_policy)    # 使用Adam优化器
        self.optimizer_value = torch.optim.Adam(self.v_net.parameters(), lr=lr_value)           # 使用Adam优化器
        self.gamma = gamma
        self.device = device

    def take_action(self, state):  
        # 根据动作概率分布随机采样
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        state = state.unsqueeze(0)
        probs = self.policy_net(state).squeeze()
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        G, returns = 0, []
        for reward in reversed(transition_dict['rewards']):
            G = self.gamma * G + reward
            returns.insert(0, G)

        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device).squeeze()     # (bsz, )
        returns = torch.tensor(returns, dtype=torch.float).view(-1, 1).to(self.device).squeeze()                        # (bsz, )
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)                   # (bsz, state_dim)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)                                  # (bsz, action_dim)

        # 梯度清零
        self.optimizer_value.zero_grad()
        self.optimizer_policy.zero_grad()
        
        # 更新价值网络
        value_predicts = self.v_net(states).squeeze()                   # (bsz, )
        value_loss = torch.mean(F.mse_loss(value_predicts, returns))     
        value_loss.backward()
        self.optimizer_value.step()

        # 更新策略网络, 从轨迹最后一步起往前计算 return，每步回传累计梯度 
        for i in reversed(range(len(rewards))):
            action = actions[i]
            state = states[i]
            value = self.v_net(state).squeeze()                         # 使用更新过的价值网络预测价值
            G = returns[i]                                              # (state_dim, )
            probs = self.policy_net(state.unsqueeze(0)).squeeze()       # (action_range, )
            log_prob = torch.log(probs[action])
            policy_loss = -log_prob * (G - value.detach())              # value 是 v_net 给出的，将其 detach 以确保只更新 policy 参数
            policy_loss.backward()        
        self.optimizer_policy.step()                                           


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

    state_dim = 4                               # 环境观测维度
    action_range = 2                            # 环境动作空间大小
    lr_policy = 2e-3
    lr_value = 3e-3
    num_episodes = 500
    hidden_dim = 64
    gamma = 0.98
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # build environment
    env_name = 'CartPole-v0'
    env = gym.make(env_name, render_mode='rgb_array')
    check_env(env.unwrapped)    # 检查环境是否符合 gym 规范
    set_seed(env, 42)

    # build agent
    agent = REINFORCE_Baseline(state_dim, hidden_dim, action_range, lr_policy, lr_value, gamma, device)

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
                    'rewards': [],
                    'dones': []
                }
                state, _ = env.reset()

                # 以当前策略交互得到一条轨迹
                while True:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(terminated or truncated)
                    state = next_state
                    episode_return += reward
                    
                    if terminated or truncated:
                        env.render()
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
            
                