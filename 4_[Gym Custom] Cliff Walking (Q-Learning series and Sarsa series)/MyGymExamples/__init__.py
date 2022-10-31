from MyGymExamples.envs.GridWorld import CliffWalkingEnv
from MyGymExamples.wrappers.HashPosition import HashPosition
from gym.envs.registration import register

register(
    id='MyGymExamples/CliffWalkingEnv-v0',
    entry_point='MyGymExamples.envs:CliffWalkingEnv',
    max_episode_steps=300,
)