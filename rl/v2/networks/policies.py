
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import *

from rl import utils
from rl.layers import preprocessing
from rl.layers.distributions import Distribution

from rl.v2.agents import Agent
from rl.v2.networks import Network, backbones

from typing import Dict, Optional, Union


# TODO: test PPO, VPG
@Network.register(name='PolicyNetwork')
class PolicyNetwork(Network):

    def __init__(self, agent: Agent, eps=utils.TF_EPS, log_prefix='policy', **kwargs):
        self._base_model_initialized = True  # weird hack
        self.eps = eps

        super().__init__(agent, target=False, log_prefix=log_prefix, **kwargs)

        self.distribution: Distribution = self.layers[-1]

    @tf.function
    def call(self, inputs, actions: Optional[utils.TensorDictOrTensor] = None, training=False, **kwargs):
        new_actions = super().call(inputs, training=training, **kwargs)

        if isinstance(actions, dict) or tf.is_tensor(actions):
            return self.distribution.log_prob(actions), self.distribution.entropy()

        return new_actions, self.distribution.log_prob(new_actions), self.distribution.mean(), self.distribution.stddev()

    def structure(self, inputs: Dict[str, Input], name='PolicyNetwork', **kwargs) -> tuple:
        # inputs = inputs['state']
        # out_args = kwargs.pop('output', kwargs.pop('distribution', {}))
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
        # output = self.output_layer(x, **out_args)
        # return inputs, output, name

        kwargs['output'] = kwargs.get('output', kwargs.pop('distribution', {}))
        return super().structure(inputs, name=name, **kwargs)

    def output_layer(self, layer: Layer, **kwargs) -> Layer:
        return Distribution.get(action_space=self.agent.env.action_space, **kwargs)(layer)

    # @tf.function
    # def objective(self, batch, reduction=tf.reduce_mean) -> tuple:
    #     advantages = batch['advantage']
    #     log_prob, entropy = self(batch['state'], actions=batch['action'], training=True)
    #
    #     # Entropy
    #     entropy = reduction(entropy)
    #     entropy_loss = entropy * self.agent.entropy_strength()
    #
    #     # Loss
    #     policy_loss = -reduction(log_prob * advantages)
    #     total_loss = policy_loss - entropy_loss
    #
    #     # Debug
    #     debug = dict(log_prob=log_prob, entropy=entropy, loss=policy_loss, loss_entropy=entropy_loss,
    #                  loss_total=total_loss)
    #
    #     return total_loss, debug

    @tf.function
    def objective(self, batch, reduction=tf.reduce_mean) -> tuple:
        advantages = batch['advantage']
        old_log_prob = batch['log_prob']

        log_prob, entropy = self(batch['state'], actions=batch['action'], training=True)

        # Entropy
        entropy = reduction(entropy)
        entropy_loss = entropy * self.agent.entropy_strength()

        # Loss
        policy_loss = -reduction(log_prob * advantages)
        total_loss = policy_loss - entropy_loss

        # Debug
        debug = dict(log_prob=log_prob, entropy=entropy, loss=policy_loss, loss_entropy=entropy_loss,
                     loss_total=total_loss, approx_kl=self._approx_kl(old_log_prob, log_prob))

        return total_loss, debug

    @staticmethod
    def _approx_kl(old_log_prob, log_prob):
        return tf.stop_gradient(tf.reduce_mean(old_log_prob - log_prob))

    # def get_distribution_layer(self, layer: Layer, min_std=0.02, unimodal=False,
    #                            **kwargs) -> tfp.layers.DistributionLambda:
    #     """
    #     A probability distribution layer, used for sampling actions, computing the `log_prob`, `entropy` ecc.
    #
    #     :param layer: last layer of the network (e.g. actor, critic, policy networks ecc)
    #     :param min_std: minimum variance, useful for 'beta' and especially 'gaussian' to prevent NaNs.
    #     :param unimodal: only used in 'beta' to make it concave and unimodal.
    #     :param kwargs: additional argument given to tf.keras.layers.Dense.
    #     :return: tfp.layers.DistributionLambda instance.
    #     """
    #     assert min_std >= 0.0
    #     min_std = tf.constant(min_std, dtype=tf.float32)
    #
    #     # Discrete actions:
    #     if self.distribution == 'categorical':
    #         num_actions = self.agent.num_actions
    #         num_classes = self.agent.num_classes
    #
    #         logits = Dense(units=num_actions * num_classes, activation='linear', **kwargs)(layer)
    #         logits = Reshape((num_actions, num_classes), name='logits')(logits)
    #
    #         return tfp.layers.DistributionLambda(
    #             make_distribution_fn=lambda t: tfp.distributions.Categorical(logits=t))(logits)
    #
    #     # Bounded continuous 1-dimensional actions:
    #     # for activations choice refer to chapter 4 of http://proceedings.mlr.press/v70/chou17a/chou17a.pdf
    #     if self.distribution == 'beta':
    #         num_actions = self.agent.num_actions
    #
    #         if unimodal:
    #             min_std += 1.0
    #
    #         alpha = Dense(units=num_actions, activation=utils.softplus(min_std), name='alpha', **kwargs)(layer)
    #         beta = Dense(units=num_actions, activation=utils.softplus(min_std), name='beta', **kwargs)(layer)
    #
    #         return tfp.layers.DistributionLambda(
    #             make_distribution_fn=lambda t: tfp.distributions.Beta(t[0], t[1]))([alpha, beta])
    #
    #     # Unbounded continuous actions)
    #     # for activations choice see chapter 4 of http://proceedings.mlr.press/v70/chou17a/chou17a.pdf
    #     if self.distribution == 'gaussian':
    #         num_actions = self.agent.num_actions
    #
    #         mu = Dense(units=num_actions, activation='linear', name='mu', **kwargs)(layer)
    #         sigma = Dense(units=num_actions, activation=utils.softplus(min_std), name='sigma', **kwargs)(layer)
    #
    #         return tfp.layers.DistributionLambda(
    #             make_distribution_fn=lambda t: tfp.distributions.Normal(loc=t[0], scale=t[1]))([mu, sigma])


# TODO: test DDPG
@Network.register(name='DeterministicPolicyNetwork')
class DeterministicPolicyNetwork(Network):

    def __init__(self, agent: Agent, *args, log_prefix='deterministic_policy', **kwargs):
        self._base_model_initialized = True
        super().__init__(agent, *args, log_prefix=log_prefix, **kwargs)

    def structure(self, inputs: Dict[str, Input], name='DeterministicPolicyNetwork', **kwargs) -> tuple:
        # inputs = inputs['state']
        #
        # if len(inputs.shape) <= 2:
        #     x = backbones.dense(layer=inputs, **kwargs)
        # else:
        #     x = backbones.convolutional(layer=inputs, **kwargs)
        #
        # output = self.output_layer(x)
        # return inputs, output, name
        return super().structure(inputs, name=name, **kwargs)

    def output_layer(self, layer: Layer, **kwargs) -> Layer:
        action_space_type = self.agent.distribution_type
        num_actions = self.agent.num_actions

        if action_space_type == 'categorical':
            # output is discrete
            return Dense(units=self.agent.num_classes, name='action_logits', **kwargs)(layer)

        if action_space_type == 'beta':
            # output is continuous and bounded
            return Dense(units=num_actions, activation='tanh', name='actions', **kwargs)(layer)

        # output is continuous (and not bounded)
        return Dense(units=num_actions, activation='linear', name='actions', **kwargs)(layer)
