
import rl
import tensorflow as tf

from tensorflow.keras.layers import *


class DenseAttention(Layer):
    def __init__(self, units: int, activation='relu', name=None, **kwargs):
        super().__init__(name=name)

        self.kernel = Dense(units=units, activation=activation, **kwargs)
        self.alpha = rl.layers.Linear(units=1, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        super().build(input_shape)

    @tf.function
    def call(self, inputs, **kwargs):
        # compute features and "attention weights"
        features = self.kernel(inputs)  # shape: (batch_size, B, units)
        weights = tf.nn.softmax(self.alpha(inputs), axis=1)  # shape: (batch_size, B, 1)

        # the output is a sum of features weighted by the attention weights
        return tf.reduce_sum(features * weights, axis=1)  # shape: (batch_size, units)


class DuelingLayer(Layer):
    """Layer representing the Dueling-architecture introduced by Dueling-DQN"""

    def __init__(self, units: int, operator='max', name=None, **kwargs):
        assert units >= 1
        assert isinstance(operator, str) and operator.lower() in ['max', 'avg']
        super().__init__(name=name)

        self.values = Dense(units=1, **kwargs)
        self.advantages = Dense(units=int(units), **kwargs)

        if operator.lower() == 'max':
            self.operator = tf.reduce_max
        else:
            self.operator = tf.reduce_mean

        self._name = f'{self._name}-{operator.lower()}'

    def call(self, x, **kwargs):
        adv = self.advantages(x)
        return self.values(x) + (adv - self.operator(adv, axis=-1, keepdims=True))  # q-values
