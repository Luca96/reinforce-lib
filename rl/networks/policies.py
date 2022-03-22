"""Policy Networks"""

import tensorflow as tf

from tensorflow.keras.layers import *

from rl.agents import Agent
from rl.networks import Network
from rl.layers.distributions import DistributionLayer

from typing import Dict


@Network.register(name='PolicyNetwork')
class PolicyNetwork(Network):

    def __init__(self, agent: Agent, log_prefix='policy', **kwargs):
        self.init_hack()
        super().__init__(agent, log_prefix=log_prefix, **kwargs)

    @tf.function
    def call(self, inputs, actions=None, **kwargs):
        distribution = super().call(inputs, **kwargs)

        if isinstance(actions, dict) or tf.is_tensor(actions):
            log_prob = distribution.log_prob(actions)
            entropy = distribution.entropy()

            if entropy is None:
                # estimate entropy
                entropy = -tf.reduce_mean(log_prob)

            return log_prob, entropy

        return tf.identity(distribution), distribution.mean(), distribution.stddev()

    def output_layer(self, layer, **kwargs) -> tf.keras.layers.Layer:
        # TODO: missing kwargs
        distribution = DistributionLayer.get(action_space=self.agent.env.action_space)
        return distribution(layer)

    @tf.function
    def objective(self, batch, reduction=tf.reduce_mean) -> tuple:
        advantages = batch['advantage']
        log_prob, entropy = self(batch['state'], actions=batch['action'], training=True)

        # Entropy loss
        entropy = reduction(entropy)
        entropy_loss = -entropy * self.agent.entropy_strength()

        # Policy loss
        policy_loss = -reduction(advantages * log_prob)
        total_loss = policy_loss + entropy_loss

        return total_loss, dict(loss=policy_loss, log_prob=log_prob, entropy=entropy, loss_entropy=entropy_loss,
                                loss_total=total_loss)


@Network.register(name='DeterministicPolicyNetwork')
class DeterministicPolicyNetwork(Network):

    def __init__(self, agent: Agent, *args, log_prefix='deterministic_policy', **kwargs):
        self.init_hack()
        super().__init__(agent, *args, log_prefix=log_prefix, **kwargs)

    def structure(self, inputs: Dict[str, Input], name='DeterministicPolicyNetwork', **kwargs) -> tuple:
        return super().structure(inputs, name=name, **kwargs)

    def output_layer(self, layer: Layer, **kwargs) -> Layer:
        num_actions = self.agent.num_actions

        # TODO: support for other action spaces?
        # output is continuous and bounded
        return Dense(units=num_actions, activation='tanh', name='actions', **kwargs)(layer)
