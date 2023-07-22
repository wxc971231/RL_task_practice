import gym
from gym import spaces
import numpy as np
import pygame
import time

class RollingBall(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"],     # 支持的渲染模式，'rgb_array' 仅用于手动交互
                "render_fps": 500,}                         # 渲染帧率

    def __init__(self, render_mode="human", width=10, height=10, show_epi=False):
        self.max_speed = 5.0
        self.width = width
        self.height = height
        self.show_epi = show_epi
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, -self.max_speed, -self.max_speed]), 
                                            high=np.array([width, height, self.max_speed, self.max_speed]),
                                            dtype=np.float64)
        self.velocity = np.zeros(2, dtype=np.float64)
        self.mass = 0.005
        self.time_step = 0.01

        # 奖励参数
        self.rewards = {'step':-2.0, 'bounce':-10.0, 'goal':300.0}
        
        # 起止位置
        self.target_position = np.array([self.width*0.8, self.height*0.8], dtype=np.float32)
        self.start_position = np.array([width*0.2, height*0.2], dtype=np.float64)
        self.position = self.start_position.copy()

        # 渲染相关
        self.render_width = 300
        self.render_height = 300
        self.scale = self.render_width / self.width
        self.window = None

        # 用于存储滚球经过的轨迹
        self.trajectory = []

        # 渲染模式支持 'human' 或 'rgb_array'
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # 渲染模式为 render_mode == 'human' 时用于渲染窗口的组件
        self.window = None
        self.clock = None

    def _get_obs(self):
        return np.hstack((self.position, self.velocity))

    def _get_info(self):
        return {}

    def step(self, action):
        # 计算加速度
        #force = action * self.mass
        acceleration = action / self.mass

        # 更新速度和位置
        self.velocity += acceleration * self.time_step
        self.velocity = np.clip(self.velocity, -self.max_speed, self.max_speed)
        self.position += self.velocity * self.time_step

        # 计算奖励
        reward = self.rewards['step']

        # 处理边界碰撞
        reward = self._handle_boundary_collision(reward)

        # 检查是否到达目标状态
        terminated, truncated = False, False
        if self._is_goal_reached():
            terminated = True
            reward += self.rewards['goal']  # 到达目标状态的奖励

        obs, info = self._get_obs(), self._get_info()
        self.trajectory.append(obs.copy())  # 记录滚球轨迹
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # 通过 super 初始化并使用基类的 self.np_random 随机数生成器
        super().reset(seed=seed)

        # 重置滚球位置、速度、轨迹
        self.position = self.start_position.copy()
        self.velocity = np.zeros(2, dtype=np.float64)
        self.trajectory = []

        return self._get_obs(), self._get_info()

    def _handle_boundary_collision(self, reward):
        if self.position[0] <= 0:
            self.position[0] = 0
            self.velocity[0] *= -1
            reward += self.rewards['bounce']
        elif self.position[0] >= self.width:
            self.position[0] = self.width
            self.velocity[0] *= -1
            reward += self.rewards['bounce']

        if self.position[1] <= 0:
            self.position[1] = 0
            self.velocity[1] *= -1
            reward += self.rewards['bounce']
        elif self.position[1] >= self.height:
            self.position[1] = self.height
            self.velocity[1] *= -1
            reward += self.rewards['bounce']

        return reward

    def _is_goal_reached(self):
        # 检查是否到达目标状态（例如，滚球到达特定位置）
        # 这里只做了一个简单的判断，可根据需要进行修改
        distance = np.linalg.norm(self.position - self.target_position)
        return distance < 1.0  # 判断距离是否小于阈值

    def render(self):
        if self.render_mode not in ["rgb_array", "human"]:
            raise False
        self._render_frame()

    def _render_frame(self):
        canvas = pygame.Surface((self.render_width, self.render_height))
        canvas.fill((255, 255, 255))    # 背景白色

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.render_width, self.render_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # 绘制目标位置
        target_position_render = self._convert_to_render_coordinate(self.target_position)
        pygame.draw.circle(canvas, (100, 100, 200), target_position_render, 20)

        # 绘制球的位置
        ball_position_render = self._convert_to_render_coordinate(self.position)
        pygame.draw.circle(canvas, (0, 0, 255), ball_position_render, 10)

        # 绘制滚球轨迹
        if self.show_epi:
            for i in range(len(self.trajectory)-1):
                position_from = self.trajectory[i]
                position_to = self.trajectory[i+1]
                position_from = self._convert_to_render_coordinate(position_from)
                position_to = self._convert_to_render_coordinate(position_to)
                color = int(230 * (i / len(self.trajectory)))  # 根据轨迹时间确定颜色深浅
                pygame.draw.lines(canvas, (color, color, color), False, [position_from, position_to], width=3)

        # 'human' 渲染模式下会弹出窗口
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])

        # 'rgb_array' 渲染模式下画面会转换为像素 ndarray 形式返回，适用于用 CNN 进行状态观测的情况，为避免影响观测不要渲染价值颜色和策略
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.quit()

    def _convert_to_render_coordinate(self, position):
        return int(position[0] * self.scale), int(self.render_height - position[1] * self.scale)

class DiscreteActionWrapper(gym.ActionWrapper):
    ''' 将 RollingBall 环境的二维连续动作空间离散化为二维离散动作空间 '''
    def __init__(self, env, bins):
        super().__init__(env)
        bin_width = 2.0 / bins
        self.action_space = spaces.MultiDiscrete([bins, bins]) 
        self.action_mapping = {i : -1+(i+0.5)*bin_width for i in range(bins)}

    def action(self, action):
        # 用向量化函数实现高效 action 映射
        vectorized_func = np.vectorize(lambda x: self.action_mapping[x])    
        result = vectorized_func(action)
        action = np.array(result)
        return action

class FlattenActionSpaceWrapper(gym.ActionWrapper):
    ''' 将多维离散动作空间拉平成一维动作空间 '''
    def __init__(self, env):
        super(FlattenActionSpaceWrapper, self).__init__(env)
        new_size = 1
        for dim in self.env.action_space.nvec:
            new_size *= dim
        self.action_space = spaces.Discrete(new_size)
    
    def action(self, action):
        orig_action = []
        for dim in reversed(self.env.action_space.nvec):
            orig_action.append(action % dim)
            action //= dim
        orig_action.reverse()
        return np.array(orig_action)