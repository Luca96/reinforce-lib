
import numpy as np
import tensorflow as tf


from rl import utils
from rl.layers import MyLayer

from typing import Union


class MyPreprocessingLayer(MyLayer):
    pass


def get(kwargs: Union[MyPreprocessingLayer, dict, str]) -> MyPreprocessingLayer:
    """Instantiates a pre-processing layer"""
    if isinstance(kwargs, MyPreprocessingLayer):
        return kwargs  # already instantiated

    if isinstance(kwargs, dict):
        name = kwargs.pop('name', None)

    elif isinstance(kwargs, str):
        name = kwargs
        kwargs = {}
    else:
        name = None

    if name is None:
        raise ValueError('Specify a name for preprocess layer.')

    name = name.lower()
    assert name in ['clip', 'min-max', 'minmax', 'standard', 'divide']

    if name == 'clip':
        return Clip(**kwargs)

    if name == 'min-max' or name == 'minmax':
        return MinMaxScaling(**kwargs)

    if name == 'standard':
        return StandardScaler(**kwargs)

    if name == 'divide':
        return Divide(**kwargs)


class Clip(MyPreprocessingLayer):
    """Layer that clips its inputs"""
    def __init__(self, min_value: Union[int, float], max_value: Union[int, float], **kwargs):
        assert tf.reduce_all(min_value < max_value)
        super().__init__(**kwargs)

        self.clip_min = tf.constant(min_value, dtype=tf.float32)
        self.clip_max = tf.constant(max_value, dtype=tf.float32)

    def call(self, x, **kwargs):
        return tf.clip_by_value(x, self.clip_min, self.clip_max)


class MinMaxScaling(MyPreprocessingLayer):
    """Rescales the input to lie in range [-1, 1] according to given `min` and `max` values."""

    def __init__(self, min_value: Union[int, float], max_value: Union[int, float], **kwargs):
        # assert all(tf.math.is_finite(min_value)) and all(tf.math.is_finite(max_value))
        # assert tf.reduce_all(min_value < max_value)
        assert np.all(np.isfinite(min_value)) and np.all(np.isfinite(max_value))
        assert np.all(min_value < max_value)

        super().__init__(**kwargs)

        self.min_value = tf.constant(min_value, dtype=tf.float32)
        self.delta_value = tf.constant(max_value, dtype=tf.float32) - self.min_value

    def call(self, x, **kwargs):
        x -= self.min_value
        x /= self.delta_value
        return 2.0 * x - 1.0


class StandardScaler(MyPreprocessingLayer):
    """Normalizes the input to have 0-mean and unitary variance:
        - The mean and variance are updated incrementally whenever `inference=True`.
    """

    def __init__(self, eps=utils.TF_EPS, **kwargs):
        assert eps >= 0.0
        super().__init__(extra_call_kwargs=['inference'], **kwargs)

        self.eps = tf.convert_to_tensor(eps, dtype=tf.float32)
        self.count = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        self.mean: tf.Variable = None
        self.var: tf.Variable = None
        self.std: tf.Variable = None

    def call(self, inputs, inference=False, **kwargs):
        if self.kwargs.pop('inference', inference):
            self.update_statistics(inputs)  # update stats if `inference=True`

        return (inputs - self.mean) / self.std

    def build(self, input_shape):
        self.mean = tf.Variable(tf.zeros(shape=(1,) + input_shape[1:], dtype=tf.float32), trainable=False)
        self.var = tf.Variable(tf.ones_like(self.mean), trainable=False)
        self.std = tf.Variable(tf.ones_like(self.mean), trainable=False)

    @tf.function
    def update_statistics(self, x):
        # mini-batch statistics
        mean = tf.reduce_mean(x, axis=0, keepdims=True)
        var = tf.math.reduce_variance(x, axis=0, keepdims=True)
        num = x.shape[0]

        # coefficients
        alpha = self.count / (self.count + num)
        beta = num / (self.count + num)
        gamma = 1.0 / ((self.count / num) + 2.0 + (num / self.count))

        # incremental update
        self.mean.assign(value=alpha * self.mean + beta * mean)
        self.var.assign(value=alpha * self.var + beta * var + gamma * (self.mean - mean) ** 2)
        self.std.assign(value=tf.sqrt(self.var) + self.eps)
        self.count.assign_add(num)


class Divide(MyPreprocessingLayer):
    """Divides the input by a given constant"""

    def __init__(self, value: float, **kwargs):
        super().__init__(**kwargs)

        self.value = tf.constant(1.0 / value, dtype=tf.float32)

    def call(self, x, **kwargs):
        return x * self.value
