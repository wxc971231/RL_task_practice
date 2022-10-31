import matplotlib.pyplot as plt
import numpy as np
import gym
import abc

class Slover():
    def __init__(self, env:gym.Env, alpha=0.1, gamma=0.9, epsilon=0.1, seed=None):  
        self.env = env      
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_action = env.action_space.n
        self.n_observation = env.observation_space.n
        self.Q_table = np.zeros([self.n_observation, self.n_action], dtype=float)
        self.V_table = np.zeros(self.n_observation, dtype=float)
        self.greedy_policy = np.array([np.arange(self.n_action)]*self.n_observation, dtype=object)  # greedy policy，记录每个 observation 下所有最优 action 
        self.policy_is_updated = False                                                              # 当前策略是否匹配最新的 Q_table
        self.rng = np.random.RandomState(seed)                                                      # agent 使用的随机数生成器
    
    def take_action(self, observation):
        # 确保策略是匹配最新 Q_table 的
        if not self.policy_is_updated:
            self.update_policy()

        if self.rng.random() < self.epsilon:
            action = self.rng.randint(self.n_action)
        else:
            action = self.rng.choice(self.greedy_policy[observation])
            #action = self.greedy_policy[observation][0]    # 这个等价于 np.argmax(self.Q_table[observation])，速度快一点但忽略了其他最优动作

        return action
    
    # 从 Q_table 导出 greedy policy
    def update_policy(self):
        best_action_value = np.max(self.Q_table, axis=1)
        # 返回一个 ndarray 组成的列表，每个 ndarray 由对应状态下最优动作组成
        self.greedy_policy = np.array([np.argwhere(self.Q_table[i]==best_action_value[i]).flatten() for i in range(self.n_observation)], dtype=object)
        self.policy_is_updated = True

    # 用 Q_table 及其 greedy policy 计算 V_table
    def update_V_table(self):
        if not self.policy_is_updated:
            self.update_policy()

        for i in range(self.n_observation):
            greedy_actions = self.greedy_policy[i]
            self.V_table[i] = self.Q_table[i][greedy_actions[0]]    # 贪心策略有 v(s) = E_a[Q(s,a)|s] = max_a Q(s,a|s)

    @abc.abstractmethod
    def update_Q_table(self):
        pass

class QLearning(Slover):
    def __init__(self, env:gym.Env, alpha=0.1, gamma=0.9, epsilon=0.1, seed=None):
        super().__init__(env, alpha, gamma, epsilon, seed)
    
    def update_Q_table(self, s, a, r, s_):
        td_target = r + self.gamma * self.Q_table[s_].max()
        td_error = td_target - self.Q_table[s,a]
        self.Q_table[s,a] += self.alpha * td_error
        self.policy_is_updated = False

class Sarsa(Slover):
    def __init__(self, env:gym.Env, alpha=0.1, gamma=0.9, epsilon=0.1, seed=None):
        super().__init__(env, alpha, gamma, epsilon, seed)
    
    def update_Q_table(self, s, a, r, s_, a_):
        td_target = r + self.gamma * self.Q_table[s_, a_]
        td_error = td_target - self.Q_table[s,a]
        self.Q_table[s,a] += self.alpha * td_error
        self.policy_is_updated = False

class SarsaNStep(Slover):
    def __init__(self, env:gym.Env, n_step=5, alpha=0.1, gamma=0.9, epsilon=0.1, seed=None):
        super().__init__(env, alpha, gamma, epsilon, seed)
        self.n_step = n_step    # 展开的步数
        self.state_list = []    # 保存之前的状态
        self.action_list = []   # 保存之前的动作
        self.reward_list = []   # 保存之前的奖励

    def update_Q_table(self, s, a, r, s_, a_, done):
        self.state_list.append(s)
        self.action_list.append(a)
        self.reward_list.append(r)

        # 保存的数据足够了，就进行 n step sarsa 更新
        if len(self.state_list) == self.n_step:
            # 计算后 n 步的收益作为 TD target
            G = self.Q_table[s_,a_]
            for i in reversed(range(self.n_step)):
                G = self.gamma * G + self.reward_list[i]    
            
            # 对list中第一个动作状态 pair 进行更新，然后将其从 list 移除 
            s = self.state_list.pop(0)  
            a = self.action_list.pop(0)
            self.reward_list.pop(0)
            self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])

        # 运行至此 list 中存储的轨迹长度一定小于 n
        # 如果本次 update 时 (s,a) 已经到达终止状态，list 中残余的动作状态 pair 无法往后看 n 步了，就能看多少看多少，也进行更新
        if done:
            G = self.Q_table[s_,a_]
            for i in reversed(range(len(self.state_list))):
                G = self.gamma * G + self.reward_list[i]    

                s = self.state_list[i]
                a = self.action_list[i]
                self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])

            # 即将开始下一条序列，将列表全清空
            self.state_list = []  
            self.action_list = [] 
            self.reward_list = [] 

class SarsaExp(Slover):
    def __init__(self, env:gym.Env, alpha=0.1, gamma=0.9, epsilon=0.1, seed=None):
        super().__init__(env, alpha, gamma, epsilon, seed)
    
    def update_Q_table(self, s, a, r, s_):
        Q_exp = self.epsilon*self.Q_table[s_].mean() + (1-self.epsilon)*self.Q_table[s_].max()  # epsilon-greedy 策略下的 E_a[Q(s_,a)]
        td_target = r + self.gamma * Q_exp
        td_error = td_target - self.Q_table[s,a]
        self.Q_table[s,a] += self.alpha * td_error
        self.policy_is_updated = False

class QLearningDouble(Slover):
    def __init__(self, env:gym.Env, alpha=0.1, gamma=0.9, epsilon=0.1, seed=None):
        super().__init__(env, alpha, gamma, epsilon, seed)
        self.Q_table_ = self.Q_table.copy()

    def update_V_table(self):
        if not self.policy_is_updated:
            self.update_policy()

        for i in range(self.n_observation):
            Q_mean = 0.5*(self.Q_table_ + self.Q_table)
            greedy_actions = self.greedy_policy[i]
            self.V_table[i] = Q_mean[i][greedy_actions[0]]    # 贪心策略有 v(s) = E_a[Q(s,a)|s] = max_a Q(s,a|s)

    def update_policy(self):
        Q_sum = self.Q_table_ + self.Q_table
        best_action_value = np.max(Q_sum, axis=1)
        # 返回一个 ndarray 组成的列表，每个 ndarray 由对应状态下最优动作组成
        self.greedy_policy = np.array([np.argwhere(Q_sum[i]==best_action_value[i]).flatten() for i in range(self.n_observation)])
        self.policy_is_updated = True

    def update_Q_table(self, s, a, r, s_):
        if self.rng.random() < 0.5:
            td_target = r + self.gamma * self.Q_table[s_][np.argmax(self.Q_table_[s_])]
            td_error = td_target - self.Q_table_[s,a]
            self.Q_table_[s,a] += self.alpha * td_error
        else:
            td_target = r + self.gamma * self.Q_table_[s_][np.argmax(self.Q_table[s_])]
            td_error = td_target - self.Q_table[s,a]
            self.Q_table[s,a] += self.alpha * td_error       
        self.policy_is_updated = False