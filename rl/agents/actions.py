
import gym
import numpy as np
import tensorflow as tf

from gym import spaces
from typing import Dict, List, Union


# TODO: pretty print (__str__)?
class ActionConverter:
    """Translates the action(s) outputted by the agent for proper usage in the environment:
        - Usually, actions need to be scaled, converted to numpy, and/or properly cast.
    """

    def __call__(self, action):
        return self.convert(action)

    def convert(self, action):
        """Implements the logic to convert the given `action`"""
        raise NotImplementedError

    @staticmethod
    def get(action_space: gym.Space, **kwargs) -> 'ActionConverter':
        """Returns an `ActionConverter` instance from given `action_space`"""
        if isinstance(action_space, spaces.Discrete):
            return DiscreteConverter(action_space, **kwargs)

        if isinstance(action_space, spaces.MultiBinary):
            return BinaryConverter(action_space, **kwargs)

        if isinstance(action_space, spaces.MultiDiscrete):
            return MultiDiscreteConverter(action_space, **kwargs)

        if isinstance(action_space, spaces.Box):
            if not action_space.is_bounded():
                return ContinuousConverter(action_space, **kwargs)

            if kwargs.pop('squashed', False):
                return TanhConverter(action_space, **kwargs)

            return BetaConverter(action_space, **kwargs)

        if isinstance(action_space, spaces.Dict):
            # only one nesting level is supported
            converters = {}

            for key, space in action_space.spaces.items():
                # assert isinstance(space, (spaces.Discrete, spaces.Box, spaces.MultiBinary))
                converters[key] = ActionConverter.get(space, **kwargs.get(key, {}))

            return DictConverter(converters)

        raise ValueError(f'Not supported action space: "{type(action_space)}".')


class IdentityConverter(ActionConverter):
    """A dummy action converter that just returns the given action as it is."""

    def __init__(self, space: gym.Space, **kwargs):
        assert not isinstance(space, (spaces.Dict, spaces.Tuple))

        self.dtype = space.dtype
        self.shape = space.shape

    def convert(self, action):
        return np.array(action, dtype=self.dtype).reshape(self.shape)


class BinaryConverter(ActionConverter):
    """Converts actions from multi-binary space"""

    def __init__(self, space: spaces.MultiBinary, **kwargs):
        assert isinstance(space, spaces.MultiBinary)

        self.num_actions = space.n
        self.num_classes = 2

    def convert(self, action):
        action = tf.cast(tf.squeeze(action), dtype=tf.int32)
        return action.numpy()


class DiscreteConverter(ActionConverter):
    """Converts actions from discrete action-space"""

    def __init__(self, space: spaces.Discrete, **kwargs):
        assert isinstance(space, spaces.Discrete)

        self.start = space.start  # used to shift actions
        self.num_actions = 1
        self.num_classes = space.n

    def convert(self, action):
        action = tf.cast(tf.squeeze(action + self.start), dtype=tf.int32)
        return action.numpy()


class MultiDiscreteConverter(ActionConverter):
    """Converts actions from a multi-discrete space"""

    def __init__(self, space: spaces.MultiDiscrete, **kwargs):
        assert isinstance(space, spaces.MultiDiscrete)

        self.num_actions = space.shape[0]
        self.num_classes = np.max(space.nvec)

    def convert(self, action):
        action = tf.cast(tf.reshape(action, shape=(self.num_actions,)), dtype=tf.int32)
        return action.numpy()


class ContinuousConverter(ActionConverter):
    """Converts actions from continuous, unbounded action-space; also drawn from a Gaussian distribution"""

    def __init__(self, space: gym.spaces.Box, **kwargs):
        assert isinstance(space, gym.spaces.Box)

        self.num_actions = space.shape[0]
        self.shape = space.shape

    def convert(self, action):
        return np.reshape(action, newshape=self.shape)


class BetaConverter(ContinuousConverter):
    """Converts actions drawn from a Beta distribution (a in [0, 1]), for continuous, bounded, action-space"""

    def __init__(self, space: gym.spaces.Box, **kwargs):
        super().__init__(space, **kwargs)
        assert space.is_bounded()

        self.action_low = tf.constant(space.low, dtype=tf.float32)
        self.action_high = tf.constant(space.high, dtype=tf.float32)
        self.action_range = tf.constant(space.high - space.low, dtype=tf.float32)

    def convert(self, action):
        # action is in [0, 1]
        action = action * self.action_range + self.action_low
        return np.reshape(action, newshape=self.shape)


class TanhConverter(BetaConverter):
    """Converts actions being output of tanh operation (a in [-1, 1]); for continuous, bounded, action-space"""

    def convert(self, action):
        # action is in [-1, 1]
        action = (action + 1.0) / 2.0 * self.action_range + self.action_low
        return np.reshape(action, newshape=self.shape)


class DictConverter(ActionConverter):
    """Converts actions from a dictionary action space"""

    def __init__(self, converters: Dict[str, ActionConverter]):
        self.converters = converters

    def __getitem__(self, key):
        return self.converters[key]

    def items(self):
        return self.converters.items()

    def convert(self, action: dict) -> Dict[str, np.ndarray]:
        assert isinstance(action, dict)
        return {key: converter(action[key]) for key, converter in self.items()}


class ParallelConverterWrapper(ActionConverter):
    """Takes an ActionConverter and applies it on a list of actions, coming from parallel envs"""

    def __init__(self, converter: Union[ActionConverter, DictConverter]):
        self.wraps_dict = isinstance(converter, DictConverter)
        self.converter = converter

    def __getitem__(self, key):
        if self.wraps_dict:
            # `converter` is DictConverter which has `converters`, so: `self.converter.converters`
            return self.converter.converters[key]

        return self.converter[key]

    def __getattr__(self, name):
        return getattr(self.converter, name)

    def convert(self, action: Union[list, Dict[str, list]]) -> List[Union[dict, np.ndarray]]:
        if isinstance(action, dict):
            return self._convert_dict(action)

        return [self.converter(a) for a in action]

    # TODO: to support arbitrary nested action-spaces, this have to reconstruct the nested structure from a flat struct
    def _convert_dict(self, x: dict) -> List[Dict[str, np.ndarray]]:
        keys = list(x.keys())
        size = len(x[keys[0]])

        action = [{} for _ in range(size)]

        for i, keys_and_values in enumerate(zip([keys] * size, *x.values())):
            # keys_and_values = [[k1, k2, ..., kN], v1, v2, ..., vN]

            for k, v in zip(keys_and_values[0], keys_and_values[1:]):
                action[i][k] = self.converter[k].convert(v)

        return action
