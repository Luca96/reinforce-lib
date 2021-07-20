
import tensorflow as tf


from tensorflow.keras.layers.experimental.preprocessing import *

from rl import utils

from typing import Union


class MinMaxScaling(tf.keras.layers.Layer):
    """Rescales the input to lie in range [-1, 1] according to given `min` and `max` values."""

    def __init__(self, min_value: Union[int, float], max_value: Union[int, float], **kwargs):
        assert tf.reduce_all(min_value < max_value)
        super().__init__(**kwargs)

        self.min_value = tf.constant(min_value, dtype=tf.float32)
        self.delta_value = tf.constant(max_value, dtype=tf.float32) - self.min_value

    def __call__(self, x, **kwargs):
        x -= self.min_value
        x /= self.delta_value
        return 2.0 * x - 1.0


class StandardScaler(tf.keras.layers.Layer):
    """Normalizes the input to have 0-mean and unitary variance:
        - The mean and variance are updated incrementally whenever `training=True`.
    """

    def __init__(self, eps=utils.TF_EPS, **kwargs):
        assert eps >= 0.0
        super().__init__(**kwargs)

        self.eps = tf.convert_to_tensor(eps, dtype=tf.float32)
        self.count = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        self.mean: tf.Variable = None
        self.var: tf.Variable = None
        self.std: tf.Variable = None

    @tf.function
    def call(self, inputs, training=False, **kwargs):
        if training:
            self.update_statistics(inputs)

        return (inputs - self.mean) / self.std

    def build(self, input_shape):
        self.mean = tf.Variable(tf.zeros(shape=(1,) + input_shape[1:], dtype=tf.float32), trainable=False)
        self.var = tf.Variable(tf.zeros_like(self.mean), trainable=False)
        self.std = tf.Variable(tf.zeros_like(self.mean), trainable=False)

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
        self.std.assign(value=tf.sqrt(self.var + self.eps))
        self.count.assign_add(num)
