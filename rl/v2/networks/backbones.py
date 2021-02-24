"""Some pre-defined NNs architectures"""

from tensorflow.keras.layers import *

from rl import utils
from rl.layers import NoisyDense


def dense(layer: Layer, units=32, num_layers=2, activation='relu', normalization='layer', normalize_input=True,
          use_bias=True, bias_initializer='glorot_uniform', kernel_initializer='glorot_normal', dropout=0.0,
          **kwargs) -> Layer:
    """Feed-Forward Neural Network architecture with one input"""
    assert num_layers >= 1

    if normalize_input:
        x = utils.apply_normalization(layer, name=normalization)
    else:
        x = layer

    for _ in range(num_layers):
        x = Dense(units, activation=activation, use_bias=use_bias, bias_initializer=bias_initializer,
                  kernel_initializer=kernel_initializer, **kwargs)(x)

        if dropout > 0.0:
            x = Dropout(rate=dropout)(x)

        x = utils.apply_normalization(x, name=normalization)

    return x


def noisy_dense(layer: Layer, units=32, num_layers=2, activation='relu', normalization='layer', normalize_input=True,
                use_bias=True, bias_initializer='glorot_uniform', kernel_initializer='glorot_normal', dropout=0.0,
                sigma=0.5, noise='factorized', **kwargs) -> Layer:
    """Feed-Forward Noisy network"""
    assert num_layers >= 1

    if normalize_input:
        x = utils.apply_normalization(layer, name=normalization)
    else:
        x = layer

    for _ in range(num_layers):
        x = NoisyDense(units, activation=activation, use_bias=use_bias, bias_initializer=bias_initializer,
                       kernel_initializer=kernel_initializer, sigma=sigma, noise=noise, **kwargs)(x)

        if dropout > 0.0:
            x = Dropout(rate=dropout)(x)

        x = utils.apply_normalization(x, name=normalization)

    return x


def convolutional():
    """Simple Convolutional NNs architecture with one input"""
    raise NotImplementedError


def recurrent():
    raise NotImplementedError
