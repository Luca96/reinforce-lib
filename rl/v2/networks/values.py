
import tensorflow as tf

from tensorflow.keras.layers import *

from rl import utils
from rl.v2.agents import Agent
from rl.v2.networks import Network, backbones

from typing import Dict, Callable


@Network.register(name='ValueNetwork')
class ValueNetwork(Network):
    """A standard ValueNetwork that predicts values for given states"""

    def __init__(self, agent: Agent, target=False, log_prefix='value', loss='mse', huber=1.0, **kwargs):
        self.init_hack()

        if callable(loss):
            self.loss_fn = loss

        elif isinstance(loss, str):
            self.huber = tf.constant(huber, dtype=tf.float32)
            self.loss_fn = self._get_loss_fn(loss=loss.lower())

        super().__init__(agent, target=target, log_prefix=log_prefix, **kwargs)

    def structure(self, inputs: Dict[str, Input], name='ValueNetwork', **kwargs) -> tuple:
        # inputs = inputs['state']
        #
        # x = inputs
        # for args in kwargs.pop('preprocess', []):
        #     x = preprocessing.get(**args)(x)
        #
        # if len(inputs.shape) <= 2:
        #     x = backbones.dense(layer=x, **kwargs)
        # else:
        #     x = backbones.convolutional(layer=x, **kwargs)
        #
        # output = self.output_layer(x)
        # return inputs, output, name
        return super().structure(inputs, name=name, **kwargs)

    def output_layer(self, layer: Layer, **kwargs) -> Layer:
        return Dense(units=1, activation='linear', name='value', **kwargs)(layer)

    @tf.function
    def objective(self, batch, reduction=tf.reduce_mean) -> tuple:
        states, returns = batch['state'], batch['return']
        values = self(states, training=True)

        loss = self.loss_fn(x=returns, y=values, reduction=reduction)
        debug = dict(loss=loss,
                     explained_variance=utils.tf_explained_variance(values, returns, eps=1e-3),
                     residual_variance=utils.tf_residual_variance(values, returns, eps=1e-3))

        return loss, debug

    @tf.function
    def mse_loss(self, x: tf.Tensor, y: tf.Tensor, reduction: Callable):
        return 0.5 * reduction(tf.square(x - y))

    @tf.function
    def mae_loss(self, x: tf.Tensor, y: tf.Tensor, reduction: Callable):
        return reduction(tf.math.abs(x - y))

    @tf.function
    def huber_loss(self, x: tf.Tensor, y: tf.Tensor, reduction: Callable):
        return reduction(utils.huber_loss(errors=x - y, kappa=self.huber))

    def _get_loss_fn(self, loss: str) -> Callable:
        assert loss in ['mse', 'mae', 'huber']

        if loss == 'mse':
            return self.mse_loss

        if loss == 'mae':
            return self.mae_loss

        return self.huber_loss


# TODO: test
@Network.register(name='DecomposedValueNetwork')
class DecomposedValueNetwork(ValueNetwork):
    """A ValueNetwork that predicts values (v) decomposed into bases (b) and exponents (e), such that: v = b * 10^e.
        - See: https://ieeexplore.ieee.org/abstract/document/9506673/, chapter
    """

    def __init__(self, agent: Agent, exponent_scale=6.0, target=False, log_prefix='value', normalize_loss=True,
                 **kwargs):
        self._base_model_initialized = True  # weird hack
        self.exp_scale = tf.constant(exponent_scale, dtype=tf.float32)
        self.normalize_loss = bool(normalize_loss)

        super().__init__(agent, target=target, log_prefix=log_prefix, **kwargs)

    def output_layer(self, layer: Layer, **kwargs) -> Layer:
        base = Dense(units=1, activation=tf.nn.tanh, name='v-base', **kwargs)(layer)
        exp = Dense(units=1, activation=lambda x: self.exp_scale * tf.nn.sigmoid(x), name='v-exp',
                    **kwargs)(layer)

        return concatenate([base, exp], axis=1)

    def structure(self, inputs: Dict[str, Input], name='DecomposedValueNetwork', **kwargs) -> tuple:
        return super().structure(inputs, name=name, **kwargs)

    @tf.function
    def objective(self, batch, reduction=tf.reduce_mean) -> tuple:
        states, returns = batch['state'], batch['return']
        values = self(states, training=True)

        base_loss = 0.5 * reduction(tf.square(returns[:, 0] - values[:, 0]))
        exp_loss = 0.5 * reduction(tf.square(returns[:, 1] - values[:, 1]))

        if self.normalize_loss:
            loss = 0.25 * base_loss + exp_loss / (self.exp_scale ** 2)
        else:
            loss = base_loss + exp_loss

        return loss, dict(loss_base=base_loss, loss_exp=exp_loss,  loss=loss)
