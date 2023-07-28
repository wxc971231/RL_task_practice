import os
import sys
import gym
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import time
from gym.utils.env_checker import check_env

env_name = 'CartPole-v0'
env = gym.make(env_name, render_mode='human')
check_env(env.unwrapped)    # 检查环境是否符合 gym 规范
env.action_space.seed(10)
observation, _ = env.reset(seed=10)

# 测试环境
for i in range(100):
    while True:
        action = env.action_space.sample()
        state, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            env.reset()
            break

        time.sleep(0.01)
        env.render()

# 关闭环境渲染
env.close()