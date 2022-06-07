
import gym
import numpy as np

from rl import utils


class GymEnv(gym.Env):
    """Basic gym environment with random generator, meant for base of custom envs"""

    def __init__(self, seed=utils.GLOBAL_SEED):
        self.random: np.random.Generator = None
        self._seed = None
        self.seed(seed)

    def sample_action(self):
        return self.action_space.sample()

    def seed(self, seed=utils.GLOBAL_SEED):
        self._seed = seed
        self.random = utils.get_random_generator(seed=self._seed)


class DiscreteWrapper(gym.ObservationWrapper, gym.ActionWrapper):
    """Discretize the state and action space of a given gym.Env instance"""

    def __init__(self, env: gym.Env, state_bins=10, action_bins=10, state_range: tuple = None,
                 action_range: tuple = None):
        assert isinstance(env, gym.Env)
        assert state_bins >= 2
        assert action_bins >= 2

        super().__init__(env)

        # init observation-space binning
        if isinstance(self.observation_space, gym.spaces.Box):
            self.should_bin_states = True

            if self.observation_space.is_bounded():
                self.state_bins = np.linspace(self.observation_space.low, self.observation_space.high,
                                              num=int(state_bins))
            elif isinstance(state_range, tuple):
                self.state_bins = np.linspace(float(state_range[0]), float(state_range[1]), num=int(state_bins))
            else:
                raise ValueError('`state_range` must be a tuple when the `observation_space` is not bounded.')
        else:
            assert isinstance(self.observation_space, gym.spaces.Discrete)
            self.should_bin_states = False

        # init action-space binning
        if isinstance(self.action_space, gym.spaces.Box):
            self.should_bin_actions = True

            if self.action_space.is_bounded():
                self.action_bins = np.linspace(self.action_space.low, self.action_space.high, num=int(action_bins))

            elif isinstance(action_range, tuple):
                self.action_bins = np.linspace(float(action_range[0]), float(action_range[1]), num=int(action_bins))
            else:
                raise ValueError('`action_range` must be a tuple when the `action_space` is not bounded.')
        else:
            assert isinstance(self.action_space, gym.spaces.Discrete)
            self.should_bin_actions = False

    def observation(self, observation):
        if self.should_bin_states:
            return np.digitize(observation, bins=self.state_bins)

        return observation

    def action(self, action):
        if self.should_bin_actions:
            return np.digitize(action, bins=self.action_bins)

        return action

    def reverse_action(self, action):
        raise NotImplementedError
