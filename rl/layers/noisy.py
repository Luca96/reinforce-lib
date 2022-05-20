"""Noisy Networks for Exploration"""

import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import regularizers

from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers

from rl import utils


class NoisyDense(Layer):
    """Noisy variant of a Dense layer"""

    # TODO: add `regularizer` and `constraint`? only for mu, sigma or w and b also?
    def __init__(self, units: int, sigma=0.5, activation=None, use_bias=True, noise='factorized', **kwargs):
        utils.remove_keys(kwargs, keys=['kernel_initializer', 'bias_initializer'])
        assert noise.lower() in ['independent', 'factorized']

        super().__init__(activity_regularizer=None, **kwargs)

        self.units = int(units)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.noise_type = noise.lower()
        self.sigma_zero = float(sigma)

        self.mu_init = None
        self.sigma_init = None

        # parameters:
        self.mu_weight = None
        self.mu_bias = None
        self.sigma_weight = None
        self.sigma_bias = None

        self.weight_shape = None
        self.bias_shape = None

    def build(self, input_shape):
        self.weight_shape = (input_shape[-1], self.units)
        self.bias_shape = (self.units,)

        # initializer:
        if self.noise_type == 'independent':
            p = tf.sqrt(3.0 / input_shape[-1])

            self.mu_init = initializers.random_uniform(minval=-p, maxval=p)
            self.sigma_init = initializers.constant(value=0.017)
        else:
            # factorized
            p = tf.sqrt(float(input_shape[-1]))

            self.mu_init = initializers.random_uniform(minval=-1.0 / p, maxval=1.0 / p)
            self.sigma_init = initializers.constant(value=self.sigma_zero / p)

        # parameters for weight (W):
        self.mu_weight = self.add_weight(
            'mu_weight',
            shape=self.weight_shape,
            initializer=self.mu_init,
            regularizer=None,
            constraint=None,
            dtype=self.dtype,
            trainable=True)

        self.sigma_weight = self.add_weight(
            'sigma_weight',
            shape=self.weight_shape,
            initializer=self.sigma_init,
            regularizer=None,
            constraint=None,
            dtype=self.dtype,
            trainable=True)

        # parameters for bias (b):
        if self.use_bias:
            self.mu_bias = self.add_weight(
                'mu_bias',
                shape=self.bias_shape,
                initializer=self.mu_init,
                regularizer=None,
                constraint=None,
                dtype=self.dtype,
                trainable=True)

            self.sigma_bias = self.add_weight(
                'sigma_bias',
                shape=self.bias_shape,
                initializer=self.sigma_init,
                regularizer=None,
                constraint=None,
                dtype=self.dtype,
                trainable=True)

        self.built = True

    # TODO: make deterministic if required?
    def call(self, inputs, **kwargs):
        # sample random variables from unit Gaussian
        noise_w, noise_b = self.sample_noise()

        # weights = mu + sigma * noise
        weights = self.mu_weight + self.sigma_weight * noise_w

        if self.use_bias:
            # bias = mu + sigma * noise
            bias = self.mu_bias + self.sigma_bias * noise_b

            z = tf.matmul(inputs, weights) + bias
        else:
            z = tf.matmul(inputs, weights)

        return self.activation(z)

    def sample_noise(self):
        """Samples noise for weights and biases"""

        if self.noise_type == 'independent':
            noise_w = tf.random.normal(self.weight_shape, stddev=1.0, seed=utils.GLOBAL_SEED)

            if self.use_bias:
                noise_b = tf.random.normal(self.bias_shape, stddev=1.0, seed=utils.GLOBAL_SEED)
            else:
                noise_b = None

            return noise_w, noise_b

        # factorized
        input_shape = (self.weight_shape[0], 1)
        output_shape = (self.units,)

        noise_p = tf.random.normal(input_shape, stddev=1.0, seed=utils.GLOBAL_SEED)
        noise_q = tf.random.normal(output_shape, stddev=1.0, seed=utils.GLOBAL_SEED)

        noise_b = self.noise_fn(noise_q)
        noise_w = self.noise_fn(noise_p) * noise_b

        return noise_w, noise_b

    @staticmethod
    def noise_fn(x):
        return tf.math.sign(x) * tf.math.sqrt(tf.math.abs(x))

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)

        if input_shape[-1] is None:
            raise ValueError(f'The innermost dimension of `input_shape` must be defined, but saw: {input_shape}.')

        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        return super().get_config()