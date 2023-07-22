import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import numpy as np
import random
import torch
import collections
import time
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gym.utils.env_checker import check_env
from environment.Env_RollingBall import RollingBall, DiscreteActionWrapper, FlattenActionSpaceWrapper
from gym.wrappers import TimeLimit

class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)        # 先进先出队列

    def add(self, state, action, reward, next_state, done):  
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), reward, np.array(next_state), done

    def size(self): 
        return len(self.buffer)

class Q_Net(torch.nn.Module):
    ''' Q 网络是一个两层 MLP, 用于 DQN 和 Double DQN '''
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        return self.fc2(x)

class VA_Net(torch.nn.Module):
    ''' VA 网络是一个两层双头 MLP, 仅用于 Dueling DQN '''
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VA_Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)   # 共享网络部分
        self.fc_A = torch.nn.Linear(hidden_dim, output_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        A = self.fc_A(F.relu(self.fc1(x)))
        V = self.fc_V(F.relu(self.fc1(x)))
        Q = V + A - A.mean().item()                         # Q值由V值和A值计算得到
        return Q

class DQN(torch.nn.Module):
    ''' DQN算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_range, lr, gamma, epsilon, target_update, device, seed=None):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_range = action_range        # action 取值范围
        self.gamma = gamma                      # 折扣因子
        self.epsilon = epsilon                  # epsilon-greedy
        self.target_update = target_update      # 目标网络更新频率
        self.count = 0                          # Q_Net 更新计数
        self.rng = np.random.RandomState(seed)  # agent 使用的随机数生成器
        self.device = device                
        
        # Q 网络
        self.q_net = Q_Net(state_dim, hidden_dim, action_range).to(device)  
        # 目标网络
        self.target_q_net = Q_Net(state_dim, hidden_dim, action_range).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        
    def max_q_value_of_given_state(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        return self.q_net(state).max().item()
        
    def take_action(self, state):  
        ''' 按照 epsilon-greedy 策略采样动作 '''
        if self.rng.random() < self.epsilon:
            action = self.rng.randint(self.action_range)
        else:
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)                             # (bsz, state_dim)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)                   # (bsz, state_dim)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(self.device)               # (bsz, act_dim)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device).squeeze()     # (bsz, )
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device).squeeze()         # (bsz, )

        q_values = self.q_net(states).gather(dim=1, index=actions).squeeze()                # (bsz, )
        max_next_q_values = self.target_q_net(next_states).max(axis=1)[0]                   # (bsz, )
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)                  # (bsz, )

        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  
        self.optimizer.zero_grad()                                                         
        dqn_loss.backward() 
        self.optimizer.step()
        
        if self.count % self.target_update == 0:
            # 按一定间隔更新 target 网络参数
            self.target_q_net.load_state_dict(self.q_net.state_dict())  
        self.count += 1

class Double_DQN(DQN):
    ''' Double DQN算法 '''        
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)                             # (bsz, state_dim)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)                   # (bsz, state_dim)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(self.device)               # (bsz, act_dim)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device).squeeze()     # (bsz, )
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device).squeeze()         # (bsz, )

        q_values = self.q_net(states).gather(dim=1, index=actions).squeeze()                # (bsz, )
        max_action = self.q_net(next_states).max(axis=1)[1]                                 # (bsz, )
        max_next_q_values = self.target_q_net(next_states).gather(dim=1, index=max_action.unsqueeze(1)).squeeze()             
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)                  # (bsz, )

        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  
        self.optimizer.zero_grad()                                                         
        dqn_loss.backward() 
        self.optimizer.step()
        
        if self.count % self.target_update == 0:
            # 按一定间隔更新 target 网络参数
            self.target_q_net.load_state_dict(self.q_net.state_dict())  
        self.count += 1

class Dueling_DQN(DQN):
    ''' Dueling DQN 算法 '''            
    def __init__(self, state_dim, hidden_dim, action_dim, action_range, lr, gamma, epsilon, target_update, device, seed=None):
        super().__init__(state_dim, hidden_dim, action_dim, action_range, lr, gamma, epsilon, target_update, device, seed)
        
        # Q 网络
        self.q_net = VA_Net(state_dim, hidden_dim, action_range).to(device)  
        # 目标网络
        self.target_q_net = VA_Net(state_dim, hidden_dim, action_range).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

if __name__ == "__main__":
    def moving_average(a, window_size):
        cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
        middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
        r = np.arange(1, window_size-1, 2)
        begin = np.cumsum(a[:window_size-1])[::2] / r
        end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
        return np.concatenate((begin, middle, end))

    def set_seed(env, seed=42):
        env.action_space.seed(seed)
        env.reset(seed=seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    state_dim = 4                               # 环境观测维度
    action_dim = 1                              # 环境动作维度
    action_bins = 10                            # 动作离散 bins 数量
    action_range = action_bins * action_bins    # 环境动作空间大小
    hidden_dim = 32
    lr = 1e-3
    num_episodes = 1000
    gamma = 0.99
    epsilon_start = 0.01
    epsilon_end = 0.001
    target_update = 1000
    buffer_size = 10000
    minimal_size = 5000
    batch_size = 128
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # build environment
    env = RollingBall(width=5, height=5, show_epi=True)    
    env = FlattenActionSpaceWrapper(DiscreteActionWrapper(env, bins=10))
    env = TimeLimit(env, 100)
    check_env(env.unwrapped)            # 检查环境是否符合 gym 规范
    set_seed(env, seed=42)              

    # build agent
    replay_buffer = ReplayBuffer(buffer_size)
    agent = DQN(state_dim, hidden_dim, action_dim, action_range, lr, gamma, epsilon_start, target_update, device)

    # 随机动作来填充 replay buffer
    state, _ = env.reset()
    while replay_buffer.size() <= minimal_size:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done=terminated or truncated)
        if terminated or truncated:
            env.render()
            state, _ = env.reset()
        #print(replay_buffer.size())

    # 开始训练
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    for i in range(20):
        with tqdm(total=int(num_episodes / 20), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 20)):
                episode_return = 0
                state, _ = env.reset()
                while True:
                    # 保存经过状态的最大Q值
                    max_q_value = agent.max_q_value_of_given_state(state) * 0.005 + max_q_value * 0.995 # 平滑处理
                    max_q_value_list.append(max_q_value)                                    
                    
                    # 选择动作移动一步
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    
                    # 更新replay_buffer
                    replay_buffer.add(state, action, reward, next_state, done=terminated or truncated)
                    
                    # 当buffer数据的数量超过一定值后,才进行Q网络训练
                    assert replay_buffer.size() > minimal_size
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)

                    state = next_state
                    episode_return += reward

                    if terminated or truncated:
                        env.render()
                        break
                    
                    #env.render()

                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
                
        #env.render()
        agent.epsilon += (epsilon_end - epsilon_start) / 10

    # show policy performence
    mv_return_list = moving_average(return_list, 29)
    episodes_list = list(range(len(return_list)))
    plt.figure(figsize=(12,8))
    plt.plot(episodes_list, return_list, label='raw', alpha=0.5)
    plt.plot(episodes_list, mv_return_list, label='moving ave')
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'{agent._get_name()} on RollingBall')
    plt.legend()
    #plt.savefig(f'./result/{agent._get_name()}.png')
    plt.show()

    # show Max Q value during training
    frames_list = list(range(len(max_q_value_list)))
    plt.plot(frames_list, max_q_value_list)
    plt.axhline(max(max_q_value_list), c='orange', ls='--')
    plt.xlabel('Frames')
    plt.ylabel('Max Q_value')
    plt.title(f'{agent._get_name()} on RollingBall')
    #plt.savefig(f'./result/{agent._get_name()}_MaxQ.png')
    plt.show()