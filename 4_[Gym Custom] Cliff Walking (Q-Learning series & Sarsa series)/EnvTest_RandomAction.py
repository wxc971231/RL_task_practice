#from MyGymExamples import CliffWalkingEnv
from gym.utils.env_checker import check_env
import numpy as np
import random
import gym

map_size = (4,12)
#env = CliffWalkingEnv(render_mode='human', map_size=map_size, pix_square_size=30)
env = gym.make('MyGymExamples:MyGymExamples/CliffWalkingEnv-v0', render_mode='human', map_size=map_size, pix_square_size=30)
print(check_env(env.unwrapped))
env.action_space.seed(42)
observation, info = env.reset(seed=42)

for _ in range(10000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())    
    
    # 随机产生状态价值和策略进行渲染
    env.render(state_values=np.random.randint(0, 10, map_size), 
                policy=np.array([np.array(random.sample(list(range(5)), random.randint(1, 5))) for _ in range(map_size[0]*map_size[1])], dtype=object))
    
    # 任务完成或失败，重置环境
    if terminated or truncated:
        print(reward, info)
        observation, info = env.reset()

env.close()
