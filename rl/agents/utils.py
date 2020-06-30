import os
import gym
import scipy.signal
import numpy as np
import tensorflow as tf

from typing import Union, List, Tuple


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


def rewards_to_go(rewards, gamma: float, normalize=False):
    returns = discount_cumsum(rewards, discount=gamma)[:-1]

    if normalize:
        returns = np_normalize(returns)

    return returns


def generalized_advantage_estimation(rewards, values, gamma: float, lambda_: float, normalize=True):
    # d_t = r_t + gamma * V_t+1 - V_t
    deltas = rewards[:-1] + tf.math.multiply(values[1:], gamma) - values[:-1]

    # Compute discount factor for each timestep, i.e. (gamma * lambda)^t
    discounts = [1.0] + [gamma * lambda_] * (len(deltas) - 1)
    discounts = tf.math.cumprod(discounts)

    # Compute normalized advantages
    advantages = tf.math.cumsum(discounts * deltas)

    if normalize:
        return tf_normalize(advantages)

    return advantages


def data_to_batches(tensors: Union[List, Tuple], batch_size: int, seed=None):
    """Transform some tensors data into a dataset of mini-batches"""
    dataset = tf.data.Dataset.from_tensor_slices(tensors)
    dataset = dataset.shuffle(buffer_size=batch_size * 4, seed=seed)
    return dataset.batch(batch_size)


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


# -------------------------------------------------------------------------------------------------
# -- File Utils
# -------------------------------------------------------------------------------------------------

def makedir(*args: str) -> str:
    """Creates a directory"""
    path = os.path.join(*args)
    os.makedirs(path, exist_ok=True)
    return path

