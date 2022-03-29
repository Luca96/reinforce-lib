"""Implicit Quantile Network (IQN) Agent"""

import tensorflow as tf

from rl import utils
from rl.agents import DQN
from rl.networks.q import ImplicitQuantileNetwork
from rl.parameters import DynamicParameter

from typing import Union


class DistortionMeasures:
    """A collection of "distortion risk measures" that alters the sampled quantiles (tau), resulting either in a
       risk-averse or risk-seeking policy.
        - To be used with IQN agent.
        - `identity` is the default distortion measure, that is risk-"neutral".
    """

    @staticmethod
    def identity(quantiles: tf.Tensor) -> tf.Tensor:
        return quantiles

    @staticmethod
    def cpw(quantiles: tf.Tensor, eta=0.71) -> tf.Tensor:
        """Cumulative Probability Weighting (CPW)
            - CPW is neither convex nor concave.
            - For small `quantiles` it is locally concave and for larger `quantiles` it becomes locally convex.
        """
        tau_eta = tf.pow(quantiles, eta)

        return tau_eta / tf.pow(tau_eta + tf.pow(1.0 - quantiles, eta), 1.0 / eta)

    @staticmethod
    def wang(quantiles: tf.Tensor, eta=0.75) -> tf.Tensor:
        inverse_cdf = utils.tf_normal_inverse_cdf(quantiles)
        return utils.tf_normal_cdf(inverse_cdf + eta)

    @staticmethod
    def pow(quantiles: tf.Tensor, eta=-2.0) -> tf.Tensor:
        """Power formula:
            - `eta < 0` => risk-averse,
            - `eta > 0` => risk-seeking
        """
        exponent = 1.0 / (1.0 + tf.abs(eta))

        if eta >= 0.0:
            return tf.pow(quantiles, exponent)

        return 1.0 - tf.pow(1.0 - quantiles, exponent)

    @staticmethod
    def cvar(quantiles: tf.Tensor, eta=0.1) -> tf.Tensor:
        """Conditional Value-at-Risk (CVaR)"""
        return tf.multiply(quantiles, eta)

    @staticmethod
    def norm(quantiles: tf.Tensor, eta=3) -> tf.Tensor:
        """Norm(eta) averages the quantiles `eta` times"""
        assert eta >= 2

        tau = tf.random.uniform(shape=(int(eta) - 1,) + quantiles.shape, maxval=0.0, seed=utils.GLOBAL_SEED)
        tau = tf.concat([tau, quantiles[None, :]], axis=0)

        return tf.reduce_mean(tau, axis=0)


# https://github.com/dannysdeng/dqn-pytorch
# https://github.com/BY571/IQN-and-Extensions
class IQN(DQN):
    # https://github.com/google/dopamine/blob/master/dopamine/agents/implicit_quantile/implicit_quantile_agent.py

    def __init__(self, *args, name='iqn-agent', lr: utils.DynamicType = 3e-4, optimizer='adam', memory_size=1024,
                 policy='e-greedy', epsilon: utils.DynamicType = 0.05, clip_norm: utils.DynamicType = 1.0,
                 update_target_network: Union[bool, int] = False, polyak: utils.DynamicType = 0.995, prioritized=True,
                 huber: utils.DynamicType = 1.0, network: dict = None, policy_samples=32, quantile_samples=8, **kwargs):
        assert policy_samples >= 1
        assert quantile_samples >= 1

        self.policy_samples = int(policy_samples)  # K, below eq. 3 in paper
        self.quantile_samples = int(quantile_samples)  # N = N', eq. 3 in paper
        self.kappa = DynamicParameter.create(value=huber)  # Huber loss cutoff (k)

        network = network or {}
        network.setdefault('cls', ImplicitQuantileNetwork)

        super().__init__(*args, name=name, lr=lr, optimizer=optimizer, memory_size=memory_size, policy=policy,
                         epsilon=epsilon, clip_norm=clip_norm, update_target_network=update_target_network,
                         polyak=polyak, double=False, dueling=False, prioritized=prioritized, network=network, **kwargs)
