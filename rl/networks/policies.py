"""Policy Networks"""

import tensorflow as tf

from tensorflow.keras.layers import Layer, Input, Dense

from rl import utils
from rl.agents import Agent
from rl.networks import Network
from rl.layers.distributions import DistributionLayer


@Network.register()
class PolicyNetwork(Network):

    def __init__(self, agent: Agent, log_prefix='policy', **kwargs):
        self.init_hack()
        super().__init__(agent, log_prefix=log_prefix, **kwargs)

    @tf.function
    def call(self, inputs, actions=None, **kwargs):
        distribution: utils.DistributionOrDict = super().call(inputs, **kwargs)

        if isinstance(actions, dict) or tf.is_tensor(actions):
            return self.log_prob_and_entropy(distribution, actions)

        return self.identity(distribution), self.mean(distribution), self.stddev(distribution), self.mode(distribution)

    def output_layer(self, layer, **kwargs) -> utils.LayerOrDict:
        distribution = DistributionLayer.get(action_space=self.agent.env.action_space, **kwargs)

        if isinstance(distribution, dict):
            # action space is Dict, so we have a dictionary of Distribution layers
            return {k: dist(layer) for k, dist in distribution.items()}

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

    def identity(self, distribution: utils.DistributionOrDict):
        """Converts the (dict) distribution to tensor(s)"""
        if isinstance(distribution, dict):
            return {k: tf.identity(v) for k, v in distribution.items()}

        return tf.identity(distribution)

    def log_prob(self, distribution: utils.DistributionOrDict, actions: utils.TensorOrDict):
        """Computes the log-probability of given `actions`. If `distribution` is a dict, then log-probs are summed
        considering each sub-distribution to be independent."""
        if not isinstance(distribution, dict):
            return distribution.log_prob(actions)

        log_probs = [dist.log_prob(actions[k]) for k, dist in distribution.items()]
        log_probs = tf.concat(log_probs, axis=-1)

        # assume distributions to be independent
        return tf.reduce_sum(log_probs, axis=-1, keepdims=True)

    def entropy(self, distribution: utils.DistributionOrDict):
        if not isinstance(distribution, dict):
            return distribution.entropy()

        entropies = [dist.entropy() for dist in distribution.values()]

        if any(x is None for x in entropies):
            return None

        entropies = tf.concat(entropies, axis=-1)

        # https://en.wikipedia.org/wiki/Joint_entropy#Less_than_or_equal_to_the_sum_of_individual_entropies
        return tf.reduce_sum(entropies, axis=-1, keepdims=True)

    def log_prob_and_entropy(self, distribution: utils.DistributionOrDict, actions: utils.TensorOrDict) -> tuple:
        if not isinstance(distribution, dict):
            log_prob = distribution.log_prob(actions)
            entropy = distribution.entropy()

            if entropy is None:
                entropy = -tf.reduce_mean(log_prob, axis=-1, keepdims=True)

            return log_prob, entropy

        log_probs = {k: dist.log_prob(actions[k]) for k, dist in distribution.items()}
        entropies = {k: dist.entropy() for k, dist in distribution.items()}

        # if there are any None entropies, estimate them from log-prob
        for k, entropy in entropies.items():
            if entropy is None:
                entropies[k] = -tf.reduce_mean(log_probs[k], axis=-1, keepdims=True)

        entropies = tf.concat(list(entropies.values()), axis=-1)
        log_probs = tf.concat(list(log_probs.values()), axis=-1)

        log_prob = tf.reduce_sum(log_probs, axis=-1, keepdims=True)
        entropy = tf.reduce_sum(entropies, axis=-1, keepdims=True)
        return log_prob, entropy

    def mean(self, distribution: utils.DistributionOrDict) -> utils.TensorOrDict:
        """Returns the mean of the given `distribution`"""
        if not isinstance(distribution, dict):
            return distribution.mean()

        return {k: dist.mean() for k, dist in distribution.items()}

    def mode(self, distribution: utils.DistributionOrDict) -> utils.TensorOrDict:
        """Returns the mode of the given `distribution`"""
        if not isinstance(distribution, dict):
            return distribution.mode()

        return {k: dist.mode() for k, dist in distribution.items()}

    def stddev(self, distribution: utils.DistributionOrDict) -> utils.TensorOrDict:
        """Returns the standard deviation of the given `distribution`"""
        if not isinstance(distribution, dict):
            return distribution.stddev()

        return {k: dist.stddev() for k, dist in distribution.items()}


@Network.register()
class DeterministicPolicyNetwork(Network):

    def __init__(self, agent: Agent, *args, log_prefix='deterministic_policy', **kwargs):
        self.init_hack()
        self.num_actions = agent.action_converter.num_actions

        super().__init__(agent, *args, log_prefix=log_prefix, **kwargs)

    def output_layer(self, layer: Layer, **kwargs) -> Layer:
        num_actions = self.num_actions

        # TODO: support for other action spaces?
        # output is continuous and bounded
        return Dense(units=num_actions, activation='tanh', name='actions', **kwargs)(layer)
