
import tensorflow as tf


from tensorflow.keras.layers.experimental.preprocessing import *

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
