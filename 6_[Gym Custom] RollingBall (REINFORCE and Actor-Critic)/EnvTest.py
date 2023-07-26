import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import numpy as np
import time
from gym.utils.env_checker import check_env
from environment.Env_RollingBall import RollingBall, DiscreteActionWrapper, FlattenActionSpaceWrapper
from gym.wrappers import TimeLimit 

env = RollingBall(render_mode='human', width=5, height=5, show_epi=True)    
env = FlattenActionSpaceWrapper(DiscreteActionWrapper(env, 5))
env = TimeLimit(env, 100)
check_env(env.unwrapped)    # 检查环境是否符合 gym 规范
env.action_space.seed(10)
observation, _ = env.reset(seed=10)

# 测试环境
for i in range(100):
    while True:
        action = env.action_space.sample()
        #action = 19
        state, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            env.reset()
            break

        time.sleep(0.01)
        env.render()

# 关闭环境渲染
env.close()