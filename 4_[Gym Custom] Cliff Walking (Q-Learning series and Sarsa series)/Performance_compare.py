import matplotlib.pyplot as plt
import numpy as np

colors = ['r', 'b', 'c', 'g', 'k', 'y', 'm']
#algorithms = ['QLearning', 'QLearning_alpha=1', 'QLearning_replay=5', 'Sarsa', 'Sarsa_alpha=1']
#algorithms = ['QLearning', 'QLearning_alpha=1', 'QLearning_replay=5']
#algorithms = ['SarsaExp', 'SarsaExp_alpha=1', 'Sarsa', 'QLearning_alpha=1', 'QLearning_replay=5' ]
#algorithms = ['QLearningDouble', 'QLearningDouble_replay=5', 'QLearning', 'QLearning_replay=5']
algorithms = ['Sarsa', '5StepTreeBackup', 'Sarsa5Step']

plt.xlabel('Episodes')
plt.ylabel('Ave Returns')
plt.title('TD methods on {}'.format('Cliff Walking'))

for algo, color in zip(algorithms, colors):
    envs_returns = np.load(f'data/{algo}'+'.npy')
    num_episodes = envs_returns.shape[1]

    ave_performance = envs_returns.mean(axis=0)                                         # 多个随机种子实验的平均性能
    filter_length = int(num_episodes/20)                                                # 滑动均值滤波长度
    ave_sliding = np.convolve(ave_performance, np.ones(filter_length)/filter_length, mode='valid')  # 一维卷积实现滑动均值滤波
    x = np.arange(0.5*filter_length,0.5*filter_length + ave_sliding.shape[0])           # 滤波后数据，显示时跳过数据不完全重叠的边际部分

    plt.plot(np.arange(envs_returns.shape[1]), ave_performance, c=color, alpha=0.2)   # 原始数据半透明显示
    plt.plot(x, ave_sliding, alpha=0.8, c=color, label=algo)                                      # 滤波后数据实现显示

plt.legend()
plt.show()