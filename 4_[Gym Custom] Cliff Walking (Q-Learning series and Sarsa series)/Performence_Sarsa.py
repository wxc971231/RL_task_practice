import matplotlib.pyplot as plt
from MyGymExamples import CliffWalkingEnv, HashPosition
from RL_Slover import Sarsa
import numpy as np
from gym.wrappers import TimeLimit
import gym

if __name__ == "__main__":
    # 使用不同随机种子重复训练
    seeds = [40, 41, 42,]
    num_test = len(seeds)

    # 创建 num_test 个并行的环境
    map_size = (4,12)
    envs = gym.vector.make('MyGymExamples:MyGymExamples/CliffWalkingEnv-v0',
                            num_envs=num_test, 
                            disable_env_checker=False,
                            asynchronous=True,
                            wrappers=lambda x: HashPosition(TimeLimit(x, max_episode_steps=100)),
                            render_mode='rgb_array', 
                            map_size=map_size, 
                            pix_square_size=30)
    
    # 并行进行使用不同 seed 的 num_test 次训练，每次分 num_period 轮训练 num_episodes 条轨迹，每轮结束打印该轮平均 return
    num_period = 10
    num_episodes = 700

    # 设置环境随机种子（由于 CliffWalkingEnv 是确定性环境，这个其实没有效果）
    observations,_ = envs.reset(seed=seeds)
    envs.action_space.seed(0)
    
    # 创建运行在各个并行环境中，使用不同随机种子的 agent    
    agents = [] 
    for env_index in range(num_test):        
        epsilon, alpha, gamma = 0.1, 0.1, 0.9
        env_type = envs.env_fns[0]()
        agents.append(Sarsa(env_type, alpha, gamma, epsilon, seeds[env_index])) 

    # 训练过程参数
    episode_returns = np.zeros(num_test, dtype=int)     # 不同并行环境当前轨迹 return
    envs_returns = np.zeros((num_test, num_episodes))   # 在不同并行环境训练得到的 num_test 个长 num_episodes 的 return 变化曲线  
    cnts = np.zeros(num_test, dtype=int)                # 各个 agent 已经交互的轨迹数量
    dones = np.zeros(num_test, dtype=bool)              # 各个 agent 是否已经交互 num_episodes 条轨迹

    # agent 串行选择初始动作
    actions = []    
    for i, agent in enumerate(agents):
        actions.append(agent.take_action(observations[i]))

    # 开始训练
    while not dones.all():
        #print(observations[0], actions[0], end=' ')
        # 并行执行动作更新环境
        next_observations, rewards, terminateds, truncateds, infos = envs.step(actions)
        
        # 更新各环境当前轨迹 return
        episode_returns += rewards

        # agent 串行更新策略
        for i, agent in enumerate(agents):
            next_action = agent.take_action(next_observations[i])

            # 检查是否有环境轨迹结束
            if not dones[i]:
                if terminateds[i] or truncateds[i]:
                    envs_returns[i][cnts[i]] = episode_returns[i]
                    episode_returns[i] = 0
                    cnts[i] += 1
                    dones[i] = cnts[i]==num_episodes
                    
                    if cnts[i] % int(num_episodes/num_period) == 0:
                        print('env {}: episode {}, return {}'.format(i, cnts[i], np.mean(envs_returns[i][cnts[i]-int(num_episodes/num_period):cnts[i]])))
                        #agent.epsilon -= epsilon/num_period # 再次设置探索概率线性衰减

                    # 这里特别注意，向量化环境在有环境副本的轨迹结束时会自动 reset
                    # 由于 CliffWalkingEnv 环境 reset 方法会把观测重置到起点，这里的 next_observations 会变成起点，不能直接用来更新价值
                    agent.update_Q_table(observations[i], actions[i], rewards[i], infos['final_observation'][i], 0)  
                else:
                    agent.update_Q_table(observations[i], actions[i], rewards[i], next_observations[i], next_action)
                
            agent.update_policy()
            actions[i] = next_action
        observations = next_observations
    
    envs.close()
        
    # 绘制 return 变化图
    np.save('data/Sarsa', envs_returns)

    plt.xlabel('Episodes')
    plt.ylabel('Ave Returns')
    plt.title('Sarsa on {}'.format('Cliff Walking'))

    ave_performance = envs_returns.mean(axis=0)                                         # 多个随机种子实验的平均性能
    filter_length = max(1, int(num_episodes/15))                                        # 滑动均值滤波长度
    ave_sliding = np.convolve(ave_performance, np.ones(filter_length)/filter_length, mode='valid')  # 一维卷积实现滑动均值滤波
    x = np.arange(0.5*filter_length,0.5*filter_length + ave_sliding.shape[0])           # 滤波后数据，显示时跳过数据不完全重叠的边际部分

    plt.plot(np.arange(envs_returns.shape[1]), ave_performance, color='b', alpha=0.2)   # 原始数据半透明显示
    plt.plot(x, ave_sliding, color='b', alpha=0.8)                                      # 滤波后数据实现显示

    plt.show()