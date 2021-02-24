
import tensorflow as tf

from tensorflow.keras.layers import *

from rl.v2.agents import Agent
from rl.v2.networks import Network, backbones

from typing import Dict


class ValueNetwork(Network):
    """A standard ValueNetwork that predicts values for given states"""

    def __init__(self, agent: Agent, target=False, log_prefix='value', **kwargs):
        super().__init__(agent, target=target, log_prefix=log_prefix, **kwargs)

    def structure(self, inputs: Dict[str, Input], name='ValueNetwork', **kwargs) -> tuple:
        inputs = inputs['state']
        x = backbones.dense(layer=inputs, **kwargs)

        output = self.output_layer(x)
        return inputs, output, name

    def output_layer(self, layer: Layer) -> Layer:
        return Dense(units=1, activation='linear', name='value')(layer)

    @tf.function
    def objective(self, batch) -> tuple:
        states, returns = batch['state'], batch['return']
        values = self(states, training=True)

        loss = 0.5 * tf.reduce_mean(tf.square(returns - values))
        return loss, dict(loss=loss)


class DecomposedValueNetwork(ValueNetwork):
    """A ValueNetwork that predicts values (v) decomposed into bases (b) and exponents (e), such that: v = b * 10^e"""

    def __init__(self, agent: Agent, exponent_scale=6.0, target=False, log_prefix='value', normalize_loss=True,
                 **kwargs):
        self._base_model_initialized = True  # weird hack
        self.exp_scale = tf.constant(exponent_scale, dtype=tf.float32)
        self.normalize_loss = bool(normalize_loss)

        super().__init__(agent, target=target, log_prefix=log_prefix, **kwargs)

    def output_layer(self, layer: Layer) -> Layer:
        base = Dense(units=1, activation=tf.nn.tanh, name='v-base')(layer)
        exp = Dense(units=1, activation=lambda x: self.exp_scale * tf.nn.sigmoid(x), name='v-exp')(layer)

        return concatenate([base, exp], axis=1)

    @tf.function
    def objective(self, batch) -> tuple:
        states, returns = batch['state'], batch['return']
        values = self(states, training=True)

        base_loss = 0.5 * tf.reduce_mean(tf.square(returns[:, 0] - values[:, 0]))
        exp_loss = 0.5 * tf.reduce_mean(tf.square(returns[:, 1] - values[:, 1]))

        if self.normalize_loss:
            loss = 0.25 * base_loss + exp_loss / (self.exp_scale ** 2)
        else:
            loss = base_loss + exp_loss

        return loss, dict(loss_base=base_loss, loss_exp=exp_loss,  loss=loss)
