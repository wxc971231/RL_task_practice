#from MyGymExamples import CliffWalkingEnv
import MyGymExamples
from gym.utils.play import play
import pygame
import gym
from gym.utils.env_checker import check_env

map_size = (4,12)
#env = CliffWalkingEnv(render_mode='rgb_array', map_size=map_size, pix_square_size=30) 
env = gym.make('MyGymExamples/CliffWalkingEnv-v0', render_mode='rgb_array', map_size=map_size, pix_square_size=30)
print(check_env(env.unwrapped)) # 检查环境是否符合 gym 规范
env.action_space.seed(42)
observation, info = env.reset(seed=42)

# env.step() 后，env.render() 前的回调函数，可用来处理刚刚 timestep 中的运行信息
def palyCallback(obs_t, obs_tp1, action, rew, terminated, truncated, info): 
    if action != 0: # 非 noop 动作
        print(rew, info)

# key-action 映射关系 
mapping = {(pygame.K_UP,): 4, 
            (pygame.K_DOWN,): 2, 
            (pygame.K_LEFT,): 3, 
            (pygame.K_RIGHT,): 1}
        
# 开始交互
play(env, keys_to_action=mapping, callback=palyCallback, fps=15, noop=0)

env.close()
