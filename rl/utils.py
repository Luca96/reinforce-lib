import os
import gym
import scipy.signal
import numpy as np
import tensorflow as tf

from gym import spaces
from typing import Union, List, Dict, Tuple


def to_tensor(x, expand_axis=0):
    x = tf.convert_to_tensor(x)
    x = tf.expand_dims(x, axis=expand_axis)
    return x


def tf_normalize(x):
    """Normalizes some tensor x to 0-mean 1-stddev"""
    # return (x - tf.math.reduce_mean(x)) / (tf.math.reduce_std(x) + np.finfo(np.float32).eps)
    return (x - tf.math.reduce_mean(x)) / tf.math.reduce_std(x)


def np_normalize(x, epsilon=np.finfo(np.float32).eps):
    return (x - np.mean(x)) / (np.std(x) + epsilon)


def discount_cumsum(x, discount: float):
    """Source: https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/ppo/core.py#L45"""
    return scipy.signal.lfilter([1.0], [1.0, float(-discount)], x[::-1], axis=0)[::-1]


def gae(rewards, values, gamma: float, lambda_: float, normalize=False):
    deltas = rewards[:-1] + tf.math.multiply(values[1:], gamma) - values[:-1]
    advantages = discount_cumsum(deltas, discount=gamma * lambda_)

    if normalize:
        advantages = tf_normalize(advantages)

    return tf.cast(advantages, dtype=tf.float32)


def rewards_to_go(rewards, discount: float, normalize=False):
    returns = discount_cumsum(rewards, discount=discount)[:-1]

    if normalize:
        returns = np_normalize(returns)

    return returns


def data_to_batches(tensors: Union[List, Tuple], batch_size: int, shuffle=False, seed=None):
    """Transform some tensors data into a dataset of mini-batches"""
    dataset = tf.data.Dataset.from_tensor_slices(tensors)
    dataset = dataset.batch(batch_size)

    if shuffle:
        return dataset.shuffle(buffer_size=batch_size, seed=seed)

    return dataset


def print_info(gym_env):
    if isinstance(gym_env, str):
        gym_env = gym.make(gym_env)

    obs_space = gym_env.observation_space
    act_space = gym_env.action_space

    # Observation space:
    if isinstance(obs_space, gym.spaces.Box):
        print(f'Observation space: {obs_space}, shape: {obs_space.shape}, bounds: {obs_space.low}, {obs_space.high}')
    else:
        print(f'Observation space: {obs_space}, n: {obs_space.n}')

    # Action space:
    if isinstance(act_space, gym.spaces.Box):
        print(f'Action space: {act_space}, shape: {act_space.shape}, bounds: {act_space.low}, {act_space.high}')
    else:
        print(f'Action space: {act_space}, n: {act_space.n}')

    print('Reward range:', gym_env.reward_range)
    print('Metadata:', gym_env.metadata)


def tf_to_scalar_shape(tensor):
    return tf.reshape(tensor, shape=[])


def assert_shapes(a, b):
    assert tf.shape(a) == tf.shape(b)


def get_input_layers(state_space: Union[Dict[str, tuple], tuple, int], layer_name='input') \
        -> Dict[str, tf.keras.layers.Input]:
    layers = dict()

    if isinstance(state_space, tuple) or isinstance(state_space, int):
        layers[layer_name] = tf.keras.layers.Input(shape=state_space, dtype=tf.float32, name=layer_name)

    elif isinstance(state_space, dict):
        for name, shape in state_space.items():
            assert isinstance(shape, tuple) or isinstance(shape, int)
            layers[name] = tf.keras.layers.Input(shape=shape, dtype=tf.float32, name=name)
    else:
        raise ValueError('state_space must be one of: Dict[Tuple or int], Tuple, or int!')

    return layers


def space_to_spec(space: gym.Space) -> Union[tuple, Dict[str, tuple]]:
    """From a gym.Space object returns its shape-specification, i.e.
         - tuple: if space is Box or Discrete
         - dict[str, tuple]: if space is Dict
    """
    if isinstance(space, spaces.Box):
        return space.shape

    if isinstance(space, spaces.Discrete):
        return space.n,   # -> tuple (space.n,)

    assert isinstance(space, spaces.Dict)

    spec = dict()
    for name, space in space.spaces.items():
        # use recursion to handle arbitrary nested Dicts
        spec[name] = space_to_spec(space)

    return spec


# -------------------------------------------------------------------------------------------------
# -- File Utils
# -------------------------------------------------------------------------------------------------

def makedir(*args: str) -> str:
    """Creates a directory"""
    path = os.path.join(*args)
    os.makedirs(path, exist_ok=True)
    return path

