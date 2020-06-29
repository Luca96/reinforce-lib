import os
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


def gae(rewards, values, gamma: float, lambda_: float, normalize=True):
    deltas = rewards[:-1] + tf.math.multiply(values[1:], gamma) - values[:-1]
    advantages = discount_cumsum(deltas, discount=gamma * lambda_)

    if normalize:
        advantages = tf_normalize(advantages)

    return tf.cast(advantages, dtype=tf.float32)


def returns(rewards, gamma: float):
    return discount_cumsum(rewards, discount=gamma)[:-1]


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


def rewards_to_go(rewards, gamma: float):
    # Discount factors for each timestep:
    discounts = np.cumprod([1.0] + [gamma] * (len(rewards) - 1))

    # Discounted cumulative sum of rewards:
    return tf.math.cumsum(discounts * np.cumsum(rewards))[:-1]


def data_to_batches(tensors: Union[List, Tuple], batch_size: int, seed=None):
    """Transform some tensors data into a dataset of mini-batches"""
    dataset = tf.data.Dataset.from_tensor_slices(tensors)
    dataset = dataset.shuffle(buffer_size=batch_size * 4, seed=seed)
    return dataset.batch(batch_size)


# -------------------------------------------------------------------------------------------------
# -- File Utils
# -------------------------------------------------------------------------------------------------

def makedir(*args: str) -> str:
    """Creates a directory"""
    path = os.path.join(*args)
    os.makedirs(path, exist_ok=True)
    return path

