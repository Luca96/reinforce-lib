
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

from rl import utils
from rl.agents import Agent
from rl.v2.networks import backbones, Network
from rl.parameters import DynamicParameter

from typing import Dict, Union


class DQNetwork(Network):

    def __init__(self, agent: Agent, target=True, log_prefix='q', **kwargs):
        super().__init__(agent, target=target, log_prefix=log_prefix, **kwargs)

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
        assert self.agent.num_actions == 1

        inputs = inputs['state']
        x = backbones.dense(layer=inputs, **kwargs)

        output = self.output_layer()(x)
        return inputs, output, name

    def output_layer(self) -> Layer:
        return Dense(units=self.agent.num_classes, name='q-values')

    @tf.function
    def objective(self, batch: dict) -> tuple:
        q_values = self(inputs=batch['state'], actions=batch['action'], training=True)
        q_targets = self.targets(batch)

        loss = tf.reduce_mean(0.5 * tf.square(q_values - q_targets))
        debug = dict(loss=loss, q_targets=q_targets, q_values=q_values)

        return loss, debug

    @tf.function
    def targets(self, batch: dict):
        """Computes target Q-values using target network"""
        rewards = batch['reward']
        q_values = self.target(inputs=batch['next_state'], training=False)

        targets = rewards * self.gamma * tf.reduce_max(q_values, axis=1, keepdims=True)
        targets = tf.where(batch['terminal'] == 0, x=rewards, y=targets)

        return tf.stop_gradient(targets)
