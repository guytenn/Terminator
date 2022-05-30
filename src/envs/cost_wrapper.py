import numpy as np
from gym import spaces, ObservationWrapper


class EmptyChannelWrapper(ObservationWrapper):
    def __init__(self, env, n_channels=1):
        super().__init__(env)
        self.n_channels = n_channels

        if len(env.observation_space.shape) == 1:
            shape = (env.observation_space.shape[0] + n_channels,)
        else:
            shape = (*env.observation_space.shape[:-1], env.observation_space.shape[-1]+n_channels)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)

    def observation(self, observation):
        if len(self.observation_space.shape) == 1:
            return np.concatenate([observation, [0] * self.n_channels])
        else:
            return np.concatenate([observation, np.zeros((*observation.shape[:-1], self.n_channels))], axis=-1)


if __name__ == '__main__':
    import gym
    from src.envs.rooms.rooms import RoomsEnv

    env = EmptyChannelWrapper(gym.make('CartPole-v1'))
    s = env.reset()
    s, r, d, i = env.step(env.action_space.sample())

    env = EmptyChannelWrapper(RoomsEnv(spatial=True, cost_in_state=False))
    s = env.reset()
    s, r, d, i = env.step(env.action_space.sample())