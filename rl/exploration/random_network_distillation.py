"""Implementation of Random Network Distillation"""

import numpy as np
import tensorflow as tf

from typing import Union, Tuple
from rl import utils
from rl.exploration import ExplorationMethod

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam


class RandomNetworkDistillation(ExplorationMethod):
    """Random Network Distillation by Y. Burda, H .Edwards et al.
       - paper: https://arxiv.org/pdf/1810.12894.pdf%20http://arxiv.org/abs/1810.12894
       - original code: https://github.com/openai/random-network-distillation/blob/master/ppo_agent.py
    """
    # TODO: use other losses than 'mse'?
    # TODO: bounded exploration bonus?
    # TODO: specific discount factor for exploration bonuses?
    # TODO: use 'max_episode_length' for buffers?
    def __init__(self, state_shape: tuple, reward_shape: int, batch_size: int, optimization_steps=1, num_layers=1):
        self.batch_size = batch_size
        self.optimization_steps = optimization_steps

        # Build target and predictor networks
        self.target_network = self.build_network(input_shape=state_shape, output_shape=reward_shape,
                                                 layers=num_layers, name='target')
        self.predictor_network = self.build_network(input_shape=state_shape, output_shape=reward_shape,
                                                    layers=num_layers, name='predictor')

        # Remember seen states so that we can use them as training data
        self.states = []
        self.targets = []

        # Prepare for training (compile)
        self.optimizer = Adam()

    @staticmethod
    def build_network(input_shape: tuple, output_shape: int, layers=1, units=4, name='RND'):
        inputs = Input(shape=input_shape)
        x = inputs

        for _ in range(layers):
            x = Dense(units, activation='relu')(x)

        outputs = Dense(units=output_shape, activation=None)(x)
        return Model(inputs, outputs, name=name)

    def bonus(self, state) -> float:
        """Returns the exploration bonus"""
        target = self.target_network(state, training=False)

        intrinsic_reward = losses.MSE(y_true=target,
                                      y_pred=self.predictor_network(state, training=False))

        # Update training dataset
        self.states.append(tf.squeeze(state))
        self.targets.append(tf.squeeze(target))

        return intrinsic_reward[0].numpy()

    def train(self, verbose=2):
        dataset = utils.data_to_batches((self.states, self.targets), batch_size=self.batch_size)
        loss_list = []

        for step, (states, targets) in enumerate(dataset):
            with tf.GradientTape() as tape:
                rewards = self.predictor_network(states, training=True)
                loss = tf.reduce_mean(losses.mean_squared_error(y_true=targets, y_pred=rewards))

            gradients = tape.gradient(loss, self.predictor_network.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.predictor_network.trainable_variables))
            loss_list.append(loss)

        print('RND-loss:', np.mean(loss_list))
        self.states.clear()
        self.targets.clear()
