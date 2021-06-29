"""Parallel environments"""

import gym
import numpy as np

from typing import Union, Dict


class ParallelEnv(gym.Env):
    """A sequential environment that wraps multiple environments in parallel"""

    def __init__(self, env, num=2, **kwargs):
        assert num >= 1
        self.num_envs = int(num)

        if callable(env):
            self.envs = [env(**kwargs) for _ in range(self.num_envs)]

        elif isinstance(env, str):
            self.envs = [gym.make(id=env, **kwargs) for _ in range(self.num_envs)]
        else:
            raise ValueError(f'Argument `env` must be a "str" or "callable" not {type(env)}.')

        base_obs_space = self.envs[0].observation_space

        if isinstance(base_obs_space, gym.spaces.Dict):
            self.complex_obs = True
            self.states_shape = {k: (self.num_envs,) + space.shape for k, space in base_obs_space.spaces.items()}
        else:
            self.complex_obs = False
            self.states_shape = (self.num_envs,) + self.observation_space.shape

        self.observation_space: gym.Space = base_obs_space
        self.action_space: gym.Space = self.envs[0].action_space

        self.rewards_shape = (self.num_envs, 1)
        self.terminals_shape = (self.num_envs, 1)

    def step(self, actions) -> tuple:
        states = self._empty_states()
        rewards = np.empty(shape=self.rewards_shape, dtype=np.float64)
        terminals = np.empty(shape=self.terminals_shape, dtype=np.bool)
        info = {}

        for j, (action, env) in enumerate(zip(actions, self.envs)):
            s, r, t, i = env.step(action)

            rewards[j] = r
            terminals[j] = t

            if self.complex_obs:
                assert isinstance(s, dict)

                for key, value in s.items():
                    states[j][key] = value
            else:
                states[j] = s

            for key, value in i.items():
                if key not in info:
                    info[key] = [value]
                else:
                    info[key].append(value)

        info = {k: np.stack(v) for k, v in info.items()}

        return states, rewards, terminals, info

    def reset(self):
        states = [env.reset() for env in self.envs]
        return np.stack(states)

    def render(self, mode='human', **kwargs):
        self.envs[0].render(mode=mode, **kwargs)

    def close(self):
        for env in self.envs:
            env.close()

    def seed(self, seed=None, same_seed=False):
        if seed is None:
            return

        if same_seed:
            for env in self.envs:
                env.seed(seed=seed)
        else:
            for i, env in enumerate(self.envs):
                env.seed(seed=seed * (i + 1))

    def _empty_states(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if self.complex_obs:
            return {k: np.empty(shape, dtype=np.float64) for k, shape in self.states_shape.items()}

        return np.empty(shape=self.states_shape, dtype=np.float64)
