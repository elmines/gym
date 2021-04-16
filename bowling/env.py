from typing import List
import numpy as np

class StackStateWrapper(object):

    def __init__(self, env, repeat_factor : int):
        self._env                         = env 
        self._state_buffer : List[object] = []
        self._repeat_factor               = repeat_factor

    def reset(self) -> object:
        true_obs           = np.array(self._env.reset())
        self._state_buffer = [true_obs] + (self._repeat_factor-1)*[np.zeros_like(true_obs)]
        return np.concatenate(self._state_buffer, axis=-1)

    def step(self, action):
        true_obs, rew, done, info = self._env.step(action)
        self._state_buffer = [true_obs] + self._state_buffer[:-1]
        return np.concatenate(self._state_buffer, axis=-1), rew, done, info

    def render(self):
        return self._env.render()
