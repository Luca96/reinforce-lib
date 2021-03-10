
import tensorflow as tf

from tensorflow.keras.layers import *

from rl import utils
from rl.agents import Agent

from rl.v2.networks import backbones, Network

from typing import Dict, Union


class QNetwork(Network):

    def __init__(self, agent: Agent, target=True, dueling=False, operator='avg', log_prefix='q', **kwargs):
        self._base_model_initialized = True

        if dueling:
            assert isinstance(operator, str) and operator.lower() in ['avg', 'max']
            self.operator = operator.lower()
            self.use_dueling = True
        else:
            self.use_dueling = False

        super().__init__(agent, target=target, dueling=dueling, operator=operator, log_prefix=log_prefix, **kwargs)

        self.gamma = self.agent.gamma

    @tf.function
    def call(self, inputs, actions=None, training=None):
        q_values = super().call(inputs, training=training)

        if tf.is_tensor(actions):
            # index q-values by given actions
            return utils.index_tensor(tensor=q_values, indices=actions)

        return q_values

    @tf.function
    def act(self, inputs):
        q_values = self(inputs)
        return tf.argmax(q_values, axis=-1)

    def structure(self, inputs: Dict[str, Input], name='Deep-Q-Network', **kwargs) -> tuple:
        utils.remove_keys(kwargs, keys=['dueling', 'operator'])

        inputs = inputs['state']
        x = backbones.dense(layer=inputs, **kwargs)

        output = self.output_layer(layer=x)
        return inputs, output, name

    def output_layer(self, layer: Layer) -> Layer:
        assert self.agent.num_actions == 1

        if self.use_dueling:
            return self.dueling_architecture(layer)

        return Dense(units=self.agent.num_classes, name='q-values', **self.output_args)(layer)

    def dueling_architecture(self, layer: Layer) -> Layer:
        num_actions = self.agent.num_actions

        # two streams (branches)
        value = Dense(units=1, name='value', **self.output_args)(layer)
        advantage = Dense(units=num_actions, name='advantage', **self.output_args)(layer)

        if self.operator == 'max':
            k = tf.reduce_max(advantage, axis=-1, keepdims=True)
        else:
            k = tf.reduce_mean(advantage, axis=-1, keepdims=True)

        q_values = value + (advantage - k)
        return q_values

    @tf.function
    def objective(self, batch: dict, reduction=tf.reduce_mean) -> tuple:
        q_values = self(inputs=batch['state'], actions=batch['action'], training=True)
        q_targets = self.targets(batch)

        loss = 0.5 * reduction(tf.square(q_values - q_targets))
        debug = dict(loss=loss, q_targets=q_targets, q_values=q_values)

        return loss, debug

    @tf.function
    def targets(self, batch: dict):
        """Computes target Q-values using target network"""
        rewards = batch['reward']
        q_values = self.target(inputs=batch['next_state'], training=False)

        targets = rewards + self.gamma * (1.0 - batch['terminal']) * tf.reduce_max(q_values, axis=1, keepdims=True)
        return tf.stop_gradient(targets)


class DoubleQNetwork(QNetwork):

    @tf.function
    def targets(self, batch: dict):
        rewards = batch['reward']
        next_states = batch['next_state']

        # double q-learning rule
        q_target = self(inputs=next_states, training=False)
        argmax_a = tf.expand_dims(tf.argmax(q_target, axis=-1), axis=-1)
        q_values = self.target(inputs=next_states, actions=argmax_a, training=False)

        targets = rewards + self.gamma * (1.0 - batch['terminal']) * q_values
        return tf.stop_gradient(targets)


# class DuelingNetwork(QNetwork):
#
#     def __init__(self, agent: Agent, target=True, operator='avg', log_prefix='dueling_q', **kwargs):
#         assert isinstance(operator, str) and operator.lower() in ['avg', 'max']
#
#         self.operator = operator.lower()
#         super().__init__(agent, target=target, log_prefix=log_prefix, **kwargs)
#
#     def structure(self, inputs: Dict[str, Input], name='Dueling-Network', **kwargs) -> tuple:
#         return super().structure(inputs, name=name, **kwargs)
#
#     def output_layer(self, layer: Layer) -> Layer:
#         num_actions = self.agent.num_actions
#
#         # two streams (branches)
#         value = Dense(units=1, **self.output_args)(layer)
#         advantage = Dense(units=num_actions, **self.output_args)(layer)
#
#         if self.operator == 'max':
#             k = tf.reduce_max(advantage, axis=-1, keepdims=True)
#         else:
#             k = tf.reduce_mean(advantage, axis=-1, keepdims=True)
#
#         q_values = value + (advantage - k)
#         return q_values
