""""Conditioning layers"""

import tensorflow as tf

from tensorflow.keras.layers import *


class ConcatConditioning(Layer):
    """Concatenation-based conditioning"""

    def __init__(self, units: int, name=None, **kwargs):
        super().__init__(name=name)

        self.concat = Concatenate()
        self.dense = Dense(units=int(units), activation='linear', **kwargs)

    def call(self, inputs: list, **kwargs):
        assert isinstance(inputs, (list, tuple))

        x = self.concat(inputs)
        x = self.dense(x)
        return x


class ScalingConditioning(Layer):
    """Multiplicative-based conditioning"""

    def __init__(self, activation='linear', name=None, **kwargs):
        super().__init__(name=name)

        self.activation = activation
        self.kwargs = kwargs

        self.dense: Dense = None
        self.multiply = Multiply()

    def build(self, input_shape):
        self.dense = Dense(units=input_shape[0][-1], activation=self.activation, **self.kwargs)

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, (list, tuple))
        assert len(inputs) == 2

        # condition input `x` on `z`
        x, z = inputs

        scaling = self.dense(z)
        return self.multiply([x, scaling])


class AffineConditioning(Layer):
    """Affine transform-based conditioning"""

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name)
        self.kwargs = kwargs

        self.dense_scale: Dense = None
        self.dense_bias: Dense = None

        self.multiply = Multiply()
        self.add = Add()

    def build(self, input_shape):
        shape, _ = input_shape

        self.dense_scale = Dense(units=shape[-1], activation='linear', **self.kwargs)
        self.dense_bias = Dense(units=shape[-1], activation='linear', **self.kwargs)

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, (list, tuple))
        assert len(inputs) == 2

        # condition input `x` on `z`
        x, z = inputs

        scale = self.dense_scale(z)
        bias = self.dense_bias(z)

        # apply affine transformation, i.e. y = scale(z) * x + bias(z)
        y = self.multiply([x, scale])
        y = self.add([y, bias])
        return y
