
import gym
import numpy as np


class CarRacingDiscrete(gym.ActionWrapper):
    """Discrete CarRacing Environment"""

    def __init__(self, bins=8):
        assert (bins >= 2) and (bins % 2 == 0)
        super().__init__(env=gym.make('CarRacing-v0'))

        # some variables useful to convert discrete actions into continuous ones
        self.low = self.action_space.low
        self.delta = (self.action_space.high - self.action_space.low) / bins

        # "fake" discrete action-space
        self.action_space = gym.spaces.MultiDiscrete([bins] * 3)

    def action(self, action):
        """Converts a given discrete action into the original continuous action-space"""
        steer = self.delta[0] * action[0] + self.low[0]
        gas = self.delta[1] * action[1] + self.low[1]
        brake = self.delta[2] * action[2] + self.low[2]

        return np.array([steer, gas, brake])

    def reverse_action(self, action):
        raise NotImplementedError
