import os
import numpy as np
import tensorflow as tf

from typing import Union, List, Tuple


def to_tensor(x, expand_axis=0):
    x = tf.convert_to_tensor(x)
    x = tf.expand_dims(x, axis=expand_axis)
    return x


def tf_normalize(x):
    """Normalizes some tensor x to 0-mean 1-stddev"""
    return (x - tf.math.reduce_mean(x)) / tf.math.reduce_std(x)


def generalized_advantage_estimation(rewards, values, gamma: float, lambda_: float):
    # d_t = r_t + gamma * V_t+1 - V_t
    # print('rewards', type(rewards[:-1]), rewards[:-1])
    # print('values', type(values[:-1]), values[:-1])
    deltas = rewards[:-1] + tf.math.multiply(values[1:], gamma) - values[:-1]

    # Compute discount factor for each timestep, i.e. (gamma * lambda)^t
    discounts = [1.0] + [gamma * lambda_] * (len(deltas) - 1)
    discounts = tf.math.cumprod(discounts)

    # Compute normalized advantages
    advantages = tf.math.cumsum(discounts * deltas)
    return tf_normalize(advantages)


def rewards_to_go(rewards, gamma: float):
    # Discount factors for each timestep:
    discounts = np.cumprod([1.0] + [gamma] * (len(rewards) - 1))

    # Discounted cumulative sum of rewards:
    return tf.math.cumsum(discounts * np.cumsum(rewards))


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

