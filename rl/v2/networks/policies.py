
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import *

from rl import utils

from rl.v2.agents import Agent
from rl.v2.networks import Network, backbones

from typing import Dict


@Network.register(name='PolicyNetwork')
class PolicyNetwork(Network):

    def __init__(self, agent: Agent, eps=utils.EPSILON, log_prefix='policy', **kwargs):
        self._base_model_initialized = True  # weird hack

        self.distribution = agent.distribution_type
        self.eps = eps

        super().__init__(agent, target=False, log_prefix=log_prefix, **kwargs)

    @tf.function
    def call(self, inputs, actions=None, training=False, **kwargs):
        policy: tfp.distributions.Distribution = super().call(inputs, training=training, **kwargs)

        new_actions = self._round_actions_if_necessary(actions=policy)
        log_prob = policy.log_prob(new_actions)

        if self.distribution != 'categorical':
            mean = policy.mean()
            std = policy.stddev()
        else:
            mean = std = 0.0

        if tf.is_tensor(actions):
            actions = self._round_actions_if_necessary(actions)

            # compute `log_prob` of given `actions`
            return policy.log_prob(actions), policy.entropy()
        else:
            return new_actions, log_prob, mean, std

    @tf.function
    def _round_actions_if_necessary(self, actions):
        if self.distribution == 'beta':
            # round samples (actions) before computing density:
            # https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Beta
            return tf.clip_by_value(actions, self.eps, 1.0 - self.eps)

        return actions

    def structure(self, inputs: Dict[str, Input], name='PolicyNetwork', **kwargs) -> tuple:
        inputs = inputs['state']
        x = backbones.dense(layer=inputs, **kwargs)

        output = self.output_layer(x)
        return inputs, output, name

    def output_layer(self, layer: Layer) -> Layer:
        return self.get_distribution_layer(layer, **self.output_args)

    @tf.function
    def objective(self, batch, reduction=tf.reduce_mean) -> tuple:
        advantages = batch['advantage']

        log_prob, entropy = self(batch['state'], actions=batch['action'], training=True)

        # Entropy
        entropy = reduction(entropy)
        entropy_loss = entropy * self.agent.entropy_strength()

        # Loss
        policy_loss = -reduction(log_prob * advantages)
        total_loss = policy_loss - entropy_loss

        # Debug
        debug = dict(log_prob=log_prob, entropy=entropy, loss=policy_loss, loss_entropy=entropy_loss,
                     loss_total=total_loss)

        return total_loss, debug

    # TODO: investigate ghostly "seed" argument in tfp.DistributionLambda
    def get_distribution_layer(self, layer: Layer, min_std=0.02, unimodal=False,
                               **kwargs) -> tfp.layers.DistributionLambda:
        """
        A probability distribution layer, used for sampling actions, computing the `log_prob`, `entropy` ecc.

        :param layer: last layer of the network (e.g. actor, critic, policy networks ecc)
        :param min_std: minimum variance, useful for 'beta' and especially 'gaussian' to prevent NaNs.
        :param unimodal: only used in 'beta' to make it concave and unimodal.
        :param kwargs: additional argument given to tf.keras.layers.Dense.
        :return: tfp.layers.DistributionLambda instance.
        """
        assert min_std >= 0.0
        min_std = tf.constant(min_std, dtype=tf.float32)

        # Discrete actions:
        if self.distribution == 'categorical':
            num_actions = self.agent.num_actions
            num_classes = self.agent.num_classes

            logits = Dense(units=num_actions * num_classes, activation='linear', **kwargs)(layer)
            logits = Reshape((num_actions, num_classes), name='logits')(logits)

            return tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Categorical(logits=t))(logits)

        # Bounded continuous 1-dimensional actions:
        # for activations choice refer to chapter 4 of http://proceedings.mlr.press/v70/chou17a/chou17a.pdf
        if self.distribution == 'beta':
            num_actions = self.agent.num_actions

            if unimodal:
                min_std += 1.0

            alpha = Dense(units=num_actions, activation=utils.softplus(min_std), name='alpha', **kwargs)(layer)
            beta = Dense(units=num_actions, activation=utils.softplus(min_std), name='beta', **kwargs)(layer)

            return tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Beta(t[0], t[1]))([alpha, beta])

        # Unbounded continuous actions)
        # for activations choice see chapter 4 of http://proceedings.mlr.press/v70/chou17a/chou17a.pdf
        if self.distribution == 'gaussian':
            num_actions = self.agent.num_actions

            mu = Dense(units=num_actions, activation='linear', name='mu', **kwargs)(layer)
            sigma = Dense(units=num_actions, activation=utils.softplus(min_std), name='sigma', **kwargs)(layer)

            return tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Normal(loc=t[0], scale=t[1]))([mu, sigma])


@Network.register(name='DeterministicPolicyNetwork')
class DeterministicPolicyNetwork(Network):

    def __init__(self, agent: Agent, *args, log_prefix='deterministic_policy', **kwargs):
        self._base_model_initialized = True
        super().__init__(agent, *args, log_prefix=log_prefix, **kwargs)

    def structure(self, inputs: Dict[str, Input], name='DeterministicPolicyNetwork', **kwargs) -> tuple:
        inputs = inputs['state']
        x = backbones.dense(layer=inputs, **kwargs)

        output = self.output_layer(x)
        return inputs, output, name

    def output_layer(self, layer: Layer) -> Layer:
        action_space_type = self.agent.distribution_type
        num_actions = self.agent.num_actions

        if action_space_type == 'categorical':
            # output is discrete
            return Dense(units=self.agent.num_classes, name='action_logits', **self.output_args)(layer)

        if action_space_type == 'beta':
            # output is continuous and bounded
            return Dense(units=num_actions, activation='tanh', name='actions', **self.output_args)(layer)

        # output is continuous (and not bounded)
        return Dense(units=num_actions, activation='linear', name='actions', **self.output_args)(layer)
