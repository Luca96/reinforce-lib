"""Dueling Architecture for DQN Agent"""

import os
import tensorflow as tf

from rl.agents.dqn import DQNAgent, DQNetwork

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


class DuelingDQNAgent(DQNAgent):
    # TODO: same tf's retracing issue as DQN
    # TODO: improve by implementing Double DQN's update rule
    def __init__(self, *args, name='dueling-dqn', **kwargs):
        load = kwargs.pop('load', False)  # prevents erroneous loading in super()

        super().__init__(*args, name=name, load=False, **kwargs)

        # adjust save path, and networks (TODO: a bit memory-inefficient for networks...)
        self.weights_path = dict(dueling_dqn=os.path.join(self.base_path, 'dueling_dqn'))
        self.dqn = DuelingNetwork(agent=self)
        self.target = DuelingNetwork(agent=self)

        if load:
            self.load()


class DuelingNetwork(DQNetwork):

    def build(self, **kwargs) -> Model:
        assert self.distribution == 'categorical'
        assert self.agent.num_actions == 1

        inputs = self._get_input_layers()
        last_layer = self.layers(inputs, **kwargs)
        value = self.value_branch(last_layer)
        advantage = self.advantage_branch(last_layer)
        q_values = tf.add(value, advantage)

        return Model(inputs, outputs=q_values, name='Dueling-Network')

    def value_branch(self, layer: Layer):
        return Dense(units=1, activation=None, name='value')(layer)

    def advantage_branch(self, layer: Layer):
        advantages = Dense(units=self.agent.num_classes, activation=None, name='advantages')(layer)
        adv_mean = tf.reduce_mean(advantages, axis=1, keepdims=True)
        return advantages - adv_mean

    def load_weights(self):
        self.net.load_weights(filepath=self.agent.weights_path['dueling_dqn'], by_name=False)

    def save_weights(self):
        self.net.save_weights(filepath=self.agent.weights_path['dueling_dqn'])
