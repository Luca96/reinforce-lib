
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


class SpatialBroadcast(Layer):
    """A layer that implements the 'spatial broadcast' operation used in VAE decoder networks.
        - Spatial Broadcast Decoder: https://arxiv.org/pdf/1901.07017
    """

    def __init__(self, width: int, height: int, **kwargs):
        w = int(width)
        h = int(height)

        assert w > 1 and h > 1
        super().__init__(**kwargs)

        # create coordinates that will later be concatenated to the tiled latents
        self.tile_shape = (1, h, w, 1)
        self.x_mesh, self.y_mesh = self.get_xy_meshgrid(w, h)

    def call(self, latents, **kwargs):
        batch_size = tf.shape(latents)[0]

        # tile the latent vectors
        z = tf.reshape(latents, shape=(batch_size, 1, 1, -1))
        z = tf.tile(z, multiples=self.tile_shape)

        # also tile the xy-meshgrid
        x = tf.tile(self.x_mesh, multiples=(batch_size, 1, 1, 1))
        y = tf.tile(self.y_mesh, multiples=(batch_size, 1, 1, 1))

        # lastly concatenate along the channel axis
        return tf.concat([z, x, y], axis=-1)

    def get_xy_meshgrid(self, w: int, h: int):
        x_coord = tf.linspace(-1, 1, w)
        y_coord = tf.linspace(-1, 1, h)

        # meshgrid & cast
        x, y = tf.meshgrid(x_coord, y_coord)
        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)

        # expand shape (to 4D) to later match the tiled latents
        x = tf.reshape(x, shape=self.tile_shape)
        y = tf.reshape(y, shape=self.tile_shape)
        return x, y
