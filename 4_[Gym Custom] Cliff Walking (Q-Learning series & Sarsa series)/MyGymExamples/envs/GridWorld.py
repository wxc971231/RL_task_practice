from typing import Tuple
import gym
from gym import spaces
import pygame
import numpy as np

class CliffWalkingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"],     # 支持的渲染模式，'rgb_array' 仅用于手动交互
                "render_fps": 500,}                         # 渲染帧率

    def __init__(self, render_mode=None, map_size=(4,12), pix_square_size=20):
        self.pix_square_size = pix_square_size  # 渲染环境中每个方格边长像素值
        self.nrow = map_size[0]
        self.ncol = map_size[1]
        self.start_location = np.array([0, self.nrow-1], dtype=int)
        self.target_location = np.array([self.ncol-1, self.nrow-1], dtype=int)

        # 观测空间
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.MultiDiscrete([self.ncol, self.nrow]),
                "target": spaces.MultiDiscrete([self.ncol, self.nrow]),
            }
        )

        # 动作空间：上下左右+noop
        self.action_space = spaces.Discrete(5)

        # 每个动作对应 agent 位置的变化
        self._action_to_direction = {
            0: np.array([0, 0]),    # noop
            1: np.array([1, 0]),    # right
            2: np.array([0, 1]),    # down
            3: np.array([-1, 0]),   # left
            4: np.array([0, -1]),   # up
        }

        # 渲染模式支持 'human' 或 'rgb_array'
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # 渲染模式为 render_mode == 'human' 时用于渲染窗口的组件
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)} # 附加信息定义为 agent 当前位置到 target 的曼哈顿距离

    def _state_transition(self, state, action):
        '''返回 agent 在 state 出执行 action 后转移到的新位置'''
        direction = self._action_to_direction[action]
        state += direction
        state[0] = np.clip(state[0], 0, self.ncol-1).item()
        state[1] = np.clip(state[1], 0, self.nrow-1).item()
        return state

    def reset(self, seed=None, options=None):
        '''step 方法给出 terminated 或 truncated 信号后，调用 reset 启动新轨迹'''
        # 通过 super 初始化并使用基类的 self.np_random 随机数生成器
        super().reset(seed=seed)

        # agent 置于起点，设置终点位置
        self._agent_location = self.start_location.copy()
        self._target_location = self.target_location.copy()

        # 获取当前状态观测和附加信息
        observation = self._get_obs()
        info = self._get_info()

        # 可以在这里刷新渲染，但我这里需要渲染最新策略，所以在测试时再手动调用 render 方法
        #if self.render_mode == "human":    
        #    self._render_frame()

        return observation, info

    def step(self, action):
        '''环境一步转移'''
        # agent 转移到执行 action 后的新位置
        self._agent_location = self._state_transition(self._agent_location, action)

        # 判断标识 terminated & truncated 并给出 reward
        terminated = np.array_equal(self._agent_location, self._target_location)
        truncated = self._agent_location[1].item() == self.nrow - 1 and self._agent_location[0].item() not in [0, self.ncol-1]
        reward = -1
        if terminated: reward = 0
        if truncated: reward = -100
        
        # 获取当前状态观测和附加信息
        observation = self._get_obs()
        info = self._get_info()

        # 可以在此刷新渲染，但本例需要渲染最新策略，所以在测试时更新策略后再手动调用 render 方法
        #if self.render_mode == "human":
        #    self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self, state_values=None, policy=None):
        if self.render_mode == "rgb_array":
            return self._render_frame()                 # 'rgb_array' 渲染模式下画面会转换为像素 ndarray 形式返回，通常用于借助 CNN 从游戏画面提取观测向量的情况，为避免影响观测不要渲染价值颜色和策略
        elif self.render_mode == "human":
            self._render_frame(state_values, policy)    # 'human' 渲染模式下会弹出窗口，如果不直接通过游戏画面提取状态观测，可以渲染价值颜色和策略，以便人员观察收敛情况
        else:
            raise False                                 # 不支持其他渲染模式，报错

    def _render_frame(self, state_values=None, policy=None):
        pix_square_size = self.pix_square_size
        canvas = pygame.Surface((self.ncol*pix_square_size, self.nrow*pix_square_size))
        canvas.fill((255, 255, 255))

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.ncol*pix_square_size, self.nrow*pix_square_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # 背景白色
        pygame.draw.rect(
            canvas,
            (255, 255, 255),
            pygame.Rect(
                (0, 0),
                (pix_square_size*self.ncol, pix_square_size*self.nrow),
            ),
        )
        # 绘制远离悬崖的方格
        if self.render_mode == "human" and isinstance(state_values, np.ndarray):
            for col in range(self.ncol):
                for row in range(self.nrow-1):
                    state_value = state_values[row][col].item()
                    max_value = 1 if np.abs(state_values).max() == 0 else np.abs(state_values).max()
                    pygame.draw.rect(
                        canvas,
                        (abs(state_value)/max_value*255, 20, 20),          # 通过颜色反映 state value
                        pygame.Rect(
                            (col*pix_square_size, row*pix_square_size),
                            (pix_square_size-1, pix_square_size-1), # 每个状态格边长减小1，这样自动出现缝线
                        ),
                    )
        else:
            for col in range(self.ncol):
                for row in range(self.nrow-1):
                    pygame.draw.rect(
                        canvas,
                        (150, 150, 150),                  
                        pygame.Rect(
                            (col*pix_square_size, row*pix_square_size),
                            (pix_square_size-1, pix_square_size-1),
                        ),
                    )
        # 绘制悬崖边最后一行方格
        for col in range(self.ncol):
            if col == 0:
                color = (100, 100, 100) # 起点
            elif col == self.ncol-1:
                color = (100, 150, 100) # 终点
            else:  
                color = (0, 0, 0)    # 悬崖
            pygame.draw.rect(
                    canvas,
                    color,
                    pygame.Rect(
                        (col*pix_square_size, (self.nrow-1)*pix_square_size),
                        (pix_square_size-1, pix_square_size-1), # 每个状态格边长减小1，这样自动出现缝线
                    ),
                )
        # 绘制 agent 
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
        # 绘制贪心策略
        if self.render_mode == "human" and isinstance(policy, np.ndarray):
            # 前几行正常行走区域
            for col in range(self.ncol):
                for row in range(self.nrow-1):
                    hash_position = col*self.nrow + row
                    actions = policy[hash_position]
                    for a in actions:
                        s_ = self._state_transition(np.array([col,row]), a)
                        if (s_ != np.array([col,row])).sum() != 0:
                            start = np.array([col*pix_square_size+0.5*pix_square_size,row*pix_square_size+0.5*pix_square_size])
                            end = s_*pix_square_size+0.5*pix_square_size
                            dot_num = 15
                            for i in range(dot_num):
                                pygame.draw.rect(
                                    canvas,
                                    (10, 255-i*175/dot_num, 10),
                                    pygame.Rect(
                                        start + (end-start) * i/dot_num,
                                        (2,2)
                                    ),
                                )
            # 最后一行只绘制起点策略
            col, row = 0, self.nrow-1
            hash_position = col*self.nrow + row
            actions = policy[hash_position]
            for a in actions:
                s_ = self._state_transition(np.array([col,row]), a)
                if (s_ != np.array([col,row])).sum() != 0:
                    start = np.array([col*pix_square_size+0.5*pix_square_size,row*pix_square_size+0.5*pix_square_size])
                    end = s_*pix_square_size+0.5*pix_square_size
                    dot_num = 15
                    for i in range(dot_num):
                        pygame.draw.rect(
                            canvas,
                            (10, 255-i*175/dot_num, 10),
                            pygame.Rect(
                                start + (end-start) * i/dot_num,
                                (2,2)
                            ),
                        )
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
            pygame.display.quit()
            pygame.quit()