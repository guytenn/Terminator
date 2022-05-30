import numpy as np
from gym import core, spaces
from src.envs.backseat_driver.unity_environment import UnityEnvironment
from src.envs.backseat_driver.unity_to_gym_wrapper import UnityToGymWrapper
from src.envs.cost_wrapper import EmptyChannelWrapper
from src.envs.base_env import BaseEnv, CostMethod
from collections import deque
import torch
import cv2
import sys
import os
import time
import copy


if sys.platform == 'darwin':
    PRESSED_KEYS_MAPPING = {3: 0, 0: 1, 1: 2}
else:
    PRESSED_KEYS_MAPPING = {82: 0, 84: 1, 83: 2}
ESC_KEY = 27


class BackseatDriver(BaseEnv):
    def __init__(self, env_path='',
                       no_graphics=True, action_repetitions=4, cost_method=CostMethod.NONE,
                       cost_coef=1., cost_in_state=True, cost_history_in_state=False,
                       cost_bias=5, window=10, termination_env=True, **kwargs
                 ):
        if termination_env:
            env_name = "BackseatDriverTerm"
        else:
            # TODO: build backseat driver env without termination
            env_name = "BackseatDriver"
        unity_env = UnityEnvironment(file_name=os.path.join(env_path, env_name),
                                     no_graphics=no_graphics,
                                     worker_id=int(time.time() * 1000) % 10000)
        self.env = EmptyChannelWrapper(UnityToGymWrapper(unity_env, False, True, False),
                                       n_channels=1+cost_in_state*cost_history_in_state*window)

        BaseEnv.__init__(self, cost_coef=cost_coef, cost_in_state=cost_in_state, cost_bias=cost_bias,
                         window=window, cost_history_in_state=cost_history_in_state, cost_method=cost_method,
                         observation_shape=self.env.observation_space.shape, n_actions=3, **kwargs)

        self.action_repetitions = action_repetitions
        self.curr_state = None

    def reset(self):
        # for _ in range(5):
        #     try:
        #         self.env.reset()
        #         self.env.step(3)
        #     except:
        #         pass
        s = self.env.reset()

        self.curr_state = s

        self.after_reset()

        return s

    def step(self, a):
        s = d = i = None
        r = 0
        coin_collected = False
        for _ in range(self.action_repetitions):
            s, ri, d, i = self.env.step(a)
            if ri >= 100:
                ri = 0
                coin_collected = True
            r += ri
            if d:
                break

        self.curr_state = s

        s, r, d, i = self.after_step(s, r, d, i, float(coin_collected))

        if i['dead']:
            try:
                self.env.step(3)
            except:
                pass

        return s, r, d, i

    def render(self, mode='human'):
        shift = int(self.cost_in_state-self.window*self.cost_in_state*self.cost_history_in_state)
        frame = self.curr_state[:, :, -3-shift:-shift]
        frame = cv2.resize(frame, dsize=(512, 256), interpolation=cv2.INTER_CUBIC)
        frame = (frame * 255).astype(np.uint8)
        if mode == 'rgb_array':
            return frame
        elif mode == 'human':
            cv2.imshow("Backseat Driver", frame)
            cv2.waitKey(10)

    def seed(self, seed=None):
        BaseEnv.seed(self, seed)
        self.env.seed(seed=seed)




if __name__ == '__main__':
    env_path = 'src/envs/backseat_driver/build/'
    env = BackseatDriver(env_path, True, cost_method=CostMethod.REAL, cost_in_state=True, cost_history_in_state=False)

    env.reset()
    while True:
        env.render()
        pressed_key = cv2.waitKey(0)
        if pressed_key == ESC_KEY:
            break
        if pressed_key in PRESSED_KEYS_MAPPING.keys():
            s, r, d, i = env.step(PRESSED_KEYS_MAPPING[pressed_key])
            print(f"aggregated cost: {env.aggregated_cost}, reward: {r}")
            if d:
                if i['dead']:
                    print('TERMINATE!')
                else:
                    print('CRASH!')
                print('-' * 20)
                env.reset()

