"""Some pre-defined NNs architectures"""

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras import regularizers

from rl import utils
from rl.layers import NoisyDense, preprocessing

from typing import Union, List, Tuple


KernelType = Union[int, Tuple[int, int], List[int], List[Tuple[int, int]]]
StrideType = KernelType


# ---------------------------------------------------------------------------------------------------------------------
# -- Feed-Forward
# ---------------------------------------------------------------------------------------------------------------------

def dense(layer: Layer, units=32, num_layers=2, activation='relu', normalization='layer', normalize_input=True,
          use_bias=True, bias_initializer='glorot_uniform', kernel_initializer='glorot_normal', dropout=0.0,
          weight_decay=0.0, **kwargs) -> Layer:
    """Feed-Forward Neural Network architecture with one input"""
    assert num_layers >= 1
    assert weight_decay >= 0.0

    if normalize_input:
        x = utils.apply_normalization(layer, name=normalization)
    else:
        x = layer

    if weight_decay > 0.0:
        weight_decay = regularizers.l2(float(weight_decay))
    else:
        weight_decay = None

    for _ in range(num_layers):
        x = Dense(units, activation=activation, use_bias=use_bias, bias_initializer=bias_initializer,
                  kernel_initializer=kernel_initializer, kernel_regularizer=weight_decay, **kwargs)(x)

        if dropout > 0.0:
            x = Dropout(rate=dropout)(x)

        x = utils.apply_normalization(x, name=normalization)

    return x


def dense_branched(*layers: List[Layer], units: Union[int, List[int]] = 32, num_layers: Union[int, List[int]] = 2,
                   activation: Union[str, List[str]] = 'relu', normalization: Union[str, List[str]] = 'layer',
                   normalize_input: Union[bool, List[bool]] = True, use_bias: Union[bool, List[bool]] = True,
                   bias_initializer='glorot_uniform', kernel_initializer='glorot_normal', weight_decay=0.0,
                   dropout: Union[float, List[float]] = 0.0, **kwargs) -> Layer:
    """Applies the `dense` backbone over each `layer` in given list of layers.
        - Returns the concatenation of each branch: i.e. concatenate([dense(layer, ...) for layer in layers]).
    """
    num_branches = len(layers)

    def init_param(param, type_):
        if isinstance(param, type_):
            return [param] * num_branches
        else:
            assert len(param) == num_branches
            return param

    units = init_param(units, int)
    num_layers = init_param(num_layers, int)
    activation = init_param(activation, str)
    normalization = [None] * num_branches if normalization is None else init_param(normalization, str)
    normalize_input = init_param(normalize_input, bool)
    use_bias = init_param(use_bias, bool)
    dropout = init_param(dropout, float)

    # apply `dense` backbone for each branch
    outputs = []

    for i in range(num_branches):
        out = dense(layer=layers[i], units=units[i], num_layers=num_layers[i], activation=activation[i],
                    normalization=normalization[i], normalize_input=normalize_input[i], use_bias=use_bias[i],
                    bias_initializer=bias_initializer, kernel_initializer=kernel_initializer, dropout=dropout[i],
                    weight_decay=weight_decay or 0.0, **kwargs)
        outputs.append(out)

    return concatenate(outputs)


# TODO: remove, use `dense(...)` with a "noisy" (bool) argument instead
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


# # TODO: implement as a layer?
# def mixture_density(layer: Layer, units: int, components: int, **kwargs) -> Layer:
#     """The "head" of a Mixture Density Network (MDN) with factored Gaussian components (called "kernel functions"):
#         - see chapter 3 of https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf
#     """
#     assert components >= 1
#     m = int(components)
#
#     alpha = Dense(units=m, activation='softmax', name='mixing_coeffs')(layer)
#
#     # Parameters of Gaussian components (note: the variance is factored)
#     mu = Dense(units, activation='linear', name='mu')(layer)
#     log_var = Dense(units, activation='linear', name='log_var')(layer)
#
#     # Compute the probability density
#     var = tf.exp(log_var)
#     pass


# ---------------------------------------------------------------------------------------------------------------------
# -- Convolutional
# ---------------------------------------------------------------------------------------------------------------------

# TODO: rename to `atari_cnn`?
def convolutional(layer: Layer, filters: Union[int, List[int]] = None, units: Union[int, List[int]] = None,
                  conv_activation='relu', dense_activation='relu', kernels: KernelType = None, resize=(84, 84),
                  strides: StrideType = None, bias_initializer='he_uniform', kernel_initializer='he_normal',
                  pooling: Union[str, dict] = None, global_pooling: str = None, padding='same', dilation=1, **kwargs):
    """Atari-like 2D-Convolutional network architecture with one input.
       Default:
         - resize images to 84x84
         - 3 convolutional layers: 32, 64, 64 filters with kernel 8, 4, 3 and strides 4, 2, 1 ("same" padding)
         - relu activation
         - flatten last conv's output
         - 1 feed-forward layer: 256 units, relu activation
    """
    num_conv = 3 if not isinstance(filters, (list, tuple)) else len(filters)
    num_dense = 1 if not isinstance(units, (list, tuple)) else len(units)

    def init_param(param, default, num=num_conv):
        if param is None:
            return default

        if isinstance(param, (int, float)):
            assert param > 0
            return [int(param)] * num

        if isinstance(param, (list, tuple)):
            assert len(param) == num
            return param

    filters = init_param(filters, default=[32, 64, 64])
    kernels = init_param(kernels, default=[8, 4, 3])
    strides = init_param(strides, default=[4, 2, 1])
    units = init_param(units, default=[256], num=num_dense)

    if isinstance(resize, tuple) and len(resize) >= 1:
        height = resize[0]
        width = height if len(resize) == 1 else resize[1]

        x = preprocessing.Resizing(height, width)(layer)
    else:
        x = layer

    for i in range(num_conv):
        x = Conv2D(filters=filters[i], kernel_size=kernels[i], strides=strides[i], padding=padding,
                   dilation_rate=dilation, activation=conv_activation, kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer, **kwargs)(x)

        x = utils.pooling_2d(layer=x, args=pooling)

    x = utils.global_pool2d_or_flatten(arg=global_pooling)(x)

    for i in range(num_dense):
        x = Dense(units=units[i], activation=dense_activation, kernel_initializer=kernel_initializer,
                  bias_initializer=bias_initializer, **kwargs)(x)
    return x


# ---------------------------------------------------------------------------------------------------------------------
# -- Recurrent
# ---------------------------------------------------------------------------------------------------------------------

def recurrent():
    raise NotImplementedError
