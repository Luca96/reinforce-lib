
import tensorflow as tf

from tensorflow.keras.layers import Layer

from rl import utils
from rl.layers import preprocessing
from rl.layers.noisy import NoisyDense
from rl.layers.conditioning import *


class Sampling(Layer):
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
