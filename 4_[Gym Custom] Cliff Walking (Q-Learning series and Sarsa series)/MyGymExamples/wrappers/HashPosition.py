
import gym

# 观测包装，把环境的原生二维观测转为一维的
class HashPosition(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        map_size = env.observation_space['agent'].nvec
        self.observation_space = gym.spaces.Discrete(map_size[0]*map_size[1]) # 新的观测空间
	
    def observation(self, obs):
        return obs["agent"][0] * self.env.nrow + obs["agent"][1]