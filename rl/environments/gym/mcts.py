
import gym
import copy
import numpy as np

from rl import utils


class MCTSEnv(gym.Env):
    """Abstract gym.Env: use to build custom environments suitable for Monte-Carlo Tree Search (MCTS).
       New methods (required by MCTS):
        - sample
        - valid_actions
        - copy
    """

    def __init__(self, *args, seed=None, **kwargs):
        self.random: np.random.Generator = None
        self._seed = None
        self.seed(seed=seed or utils.GLOBAL_SEED)

    def sample(self) -> int:
        """Returns a random but 'valid' action sampled from the environment's action space"""
        return int(self.random.choice(self.valid_actions(), size=1))

    def valid_actions(self) -> np.ndarray:
        """Returns a list of actions allowed in the current environment's state"""
        raise NotImplementedError

    def copy(self) -> 'MCTSEnv':
        return copy.deepcopy(self)

    def seed(self, seed=None):
        self._seed = seed
        self.random = utils.get_random_generator(seed=self._seed)


class MCTSWrapper(gym.Wrapper):
    """Wraps an existing gym.Env to make it compliant to Monte-Carlo Tree Search (MCTS)"""

    def __init__(self, env, seed=None):
        super().__init__(env)
        assert isinstance(self.action_space, gym.spaces.Discrete)

        self.random: np.random.Generator = None
        self._seed = None
        self.seed(seed=seed or utils.GLOBAL_SEED)

    def sample(self) -> int:
        """Returns a random but 'valid' action sampled from the environment's action space"""
        return int(self.random.choice(self.valid_actions(), size=1))

    def valid_actions(self) -> np.ndarray:
        """Returns a list of actions allowed in the current environment's state"""
        return np.arange(self.action_space.n)

    def copy(self) -> 'MCTSWrapper':
        return copy.deepcopy(self)

    def seed(self, seed=None):
        self._seed = seed
        self.random = np.random.default_rng(np.random.MT19937(seed=self._seed))



