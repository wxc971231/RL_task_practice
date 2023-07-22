from gym.envs.registration import register

register(
    id='RollingBall-v0',
    entry_point='environment.Env_RollingBall:RollingBall',
)