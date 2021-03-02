"""Parallel environments"""

import gym
import numpy as np


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

        self.observation_space: gym.Space = self.envs[0].observation_space
        self.action_space: gym.Space = self.envs[0].action_space

        self.states_shape = (self.num_envs,) + self.observation_space.shape
        self.rewards_shape = (self.num_envs, 1)
        self.terminals_shape = (self.num_envs, 1)

    # TODO(bug): if states are dictionaries => np.stack will fail
    def step(self, actions) -> tuple:
        states = np.zeros(shape=self.states_shape, dtype=np.float64)
        rewards = np.zeros(shape=self.rewards_shape, dtype=np.float64)
        terminals = np.zeros(shape=self.terminals_shape, dtype=np.bool)
        info = {}

        for j, (action, env) in enumerate(zip(actions, self.envs)):
            s, r, t, i = env.step(action)

            states[j] = s
            rewards[j] = r
            terminals[j] = t

            for k, v in i.items():
                if k not in info:
                    info[k] = [v]
                else:
                    info[k].append(v)

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
