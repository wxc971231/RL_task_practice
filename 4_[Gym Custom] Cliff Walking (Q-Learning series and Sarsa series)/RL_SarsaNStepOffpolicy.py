import matplotlib.pyplot as plt
from MyGymExamples import CliffWalkingEnv, HashPosition
from RL_Slover import SarsaNStepOffpolicy
import numpy as np
from gym.wrappers import TimeLimit
from tqdm import tqdm   # 显示循环进度条的库
import gym

# env实例
map_size = (4,12)
env = CliffWalkingEnv(render_mode='human', map_size=map_size, pix_square_size=30)
#env = gym.make('MyGymExamples/CliffWalkingEnv-v0', render_mode='human', map_size=map_size, pix_square_size=30)
np.random.seed(42)
wrapped_env = TimeLimit(env, max_episode_steps=100)
wrapped_env = HashPosition(wrapped_env)
observation, info = wrapped_env.reset(seed=42)

# agent实例
epsilon = 0.1
alpha = 0.1
gamma = 0.9
n_step = 5
agent = SarsaNStepOffpolicy(wrapped_env, n_step, alpha, gamma, epsilon, seed=42)

# 进行训练
num_episodes = 700          # 训练交互轨迹总量
num_period = 10             # 分这么多次交互完总轨迹量
return_list = []            # 记录每一条序列的回报
for i in range(num_period): # 分轮完成训练，每轮结束后统计该轮平均回报 
    with tqdm(total=int(num_episodes / num_period), desc='Iteration %d' % i) as pbar:   # tqdm的进度条功能
        for i_episode in range(int(num_episodes / num_period)):                         # 每个进度条的序列数
            episode_return = 0
            observation,_ = wrapped_env.reset()
            action = agent.take_action(observation)
            #action = wrapped_env.action_space.sample()
            wrapped_env.render(state_values=agent.V_table.reshape(-1,wrapped_env.nrow).T, policy=agent.greedy_policy)      

            while True:
                next_observation, reward, terminated, truncated, info = wrapped_env.step(action)
                next_action = agent.take_action(next_observation)
                agent.update_Q_table(observation, action, reward, next_observation, next_action, done=(terminated or truncated))
                agent.update_policy()
                episode_return += reward    # 这里回报的计算不进行折扣因子衰减
                agent.update_V_table()  

                if terminated or truncated:
                    break

                observation = next_observation
                action = next_action

            # 降低渲染频率，可以大幅提升运算速度（因为这里都是在轨迹开始时渲染，agent看起来不动）
            if i_episode % 5 == 0:
                wrapped_env.render(state_values=agent.V_table.reshape(-1,wrapped_env.nrow).T, policy=agent.greedy_policy)                    

            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)
    #agent.epsilon -= epsilon/num_period # 探索概率线性衰减

env.close()

# 绘制return变化图
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('{}-step Sarsa off-policy on {}'.format(n_step, 'Cliff Walking'))
plt.show()
