
import tensorflow as tf

from tensorflow.keras.initializers import Initializer

from rl import utils
from rl.layers import preprocessing
from rl.layers.noisy import NoisyDense
from rl.layers.conditioning import ConcatConditioning, ScalingConditioning, AffineConditioning
from rl.layers.misc import DuelingLayer

from typing import Union


class ScaledInitializer(tf.keras.initializers.Initializer):
    """Wraps a tf.keras.initializers.Initializer weight-initializer instance to rescale its output"""

    def __init__(self, scaling: float, initializer: Union[str, Initializer] = 'glorot_uniform'):
        super().__init__()

        if isinstance(initializer, str):
            self.weight_init = tf.keras.initializers.get(identifier=initializer)
        else:
            assert isinstance(initializer, Initializer)
            self.weight_init = initializer

        self.scaling = tf.constant(scaling, dtype=tf.float32)

    def __call__(self, shape, dtype=None):
        return self.scaling * self.weight_init(shape=shape)


class Sampling(tf.keras.layers.Layer):
    """Given mean and log-variance that parametrize a Gaussian, the layer samples from it by using the
       reparametrization trick.
        - https://keras.io/examples/generative/vae/
    """

    def call(self, inputs, **kwargs):
        mean, log_var = inputs

        # sample from a Standard Normal
        epsilon = tf.random.normal(shape=tf.shape(mean))

        # Reparametrization trick
        return mean + tf.exp(0.5 * log_var) * epsilon


class Linear(tf.keras.layers.Dense):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='linear', **kwargs)
