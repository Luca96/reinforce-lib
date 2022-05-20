"""Layers that wraps tfp's Probability distributions over action spaces"""

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability import layers as tfl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.layers import Layer, Dense, Reshape

from typing import Dict, Callable, Union, List, Tuple

from rl import utils
from rl.layers import MyLayer, Linear


# TODO: support for mixture distribution
class DistributionLayer(MyLayer):
    """Abstract probability distribution layer that wraps a `tfp.distribution.Distribution` instance"""

    def __init__(self, **kwargs):
        super().__init__(extra_call_kwargs=['deterministic'], **kwargs)

    def call(self, inputs, deterministic=False, **kwargs) -> tfl.DistributionLambda:
        params = self.forward_params(inputs)
        distribution = tfl.DistributionLambda(make_distribution_fn=self.make_distribution_fn,
                                              convert_to_tensor_fn=self.distribution_to_tensor(deterministic))(params)
        return distribution

    @staticmethod
    def get(action_space: gym.Space, **kwargs) -> Union['DistributionLayer', Dict[str, 'DistributionLayer']]:
        if isinstance(action_space, gym.spaces.Discrete):
            num_classes = action_space.n
            assert num_classes >= 2

            if num_classes == 2:
                return Bernoulli(num_actions=1, **kwargs)

            return Categorical(num_actions=1, num_classes=num_classes, **kwargs)

        if isinstance(action_space, gym.spaces.MultiBinary):
            # TODO: independent?
            return Bernoulli(num_actions=action_space.n, **kwargs)

        if isinstance(action_space, gym.spaces.MultiDiscrete):
            # TODO: multinomial or sample/independent categorical?
            num_actions = action_space.shape[0]
            num_classes = np.max(action_space.nvec)

            return Categorical(num_actions=num_actions, num_classes=num_classes)

        if isinstance(action_space, gym.spaces.Box):
            if action_space.is_bounded():
                return Beta(shape=action_space.shape, **kwargs)

            return Gaussian(shape=action_space.shape, **kwargs)

        assert isinstance(action_space, gym.spaces.Dict)
        distributions = {}

        for key, space in action_space.spaces.items():
            distributions[key] = DistributionLayer.get(space, **kwargs.get(key, {}))

        return distributions

    def forward_params(self, inputs) -> List[tf.Tensor]:
        raise NotImplementedError

    def make_distribution_fn(self, params) -> tfd.Distribution:
        raise NotImplementedError

    def distribution_to_tensor(self, deterministic: bool) -> Callable:
        if deterministic:
            return tfd.Distribution.mode

        return tfd.Distribution.sample


# https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/RelaxedBernoulli
class Bernoulli(DistributionLayer):

    def __init__(self, num_actions: int, **kwargs):
        assert num_actions >= 1
        super().__init__(**kwargs)

        self.num_actions = int(num_actions)
        self.logits = Dense(units=self.num_actions, name='logits', **kwargs)

    @tf.function
    def forward_params(self, inputs) -> List[tf.Tensor]:
        return self.logits(inputs)

    def make_distribution_fn(self, params) -> tfd.Bernoulli:
        return tfd.Bernoulli(logits=params)


# https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/RelaxedOneHotCategorical
class Categorical(DistributionLayer):

    class CategoricalDistribution(tfd.Categorical):
        # TODO: consider to estimate them from sampled actions
        def _mean(self, **kwargs):
            return utils.TF_ZERO

        def _stddev(self, **kwargs):
            # since it's not defined for Categorical, we return a dummy value of zero
            return utils.TF_ZERO

    def __init__(self, num_actions: int, num_classes: int, **kwargs):
        assert num_actions >= 1
        assert num_classes >= 1

        super().__init__(**kwargs)

        self.num_actions = int(num_actions)
        self.num_classes = int(num_classes)

        self.logits = Dense(units=self.num_actions * self.num_classes, **kwargs)
        self.reshape = Reshape(target_shape=(self.num_actions, self.num_classes))

    @tf.function
    def forward_params(self, inputs) -> List[tf.Tensor]:
        # TODO: support for "temperature" parameter (just divide logits by it)
        logits = self.logits(inputs)
        return self.reshape(logits)

    def make_distribution_fn(self, params) -> 'CategoricalDistribution':
        return self.CategoricalDistribution(logits=params)


class Gaussian(DistributionLayer):

    def __init__(self, shape: tuple, log_std_range=(-20, 2), **kwargs):
        assert isinstance(shape, (tuple, list))
        assert isinstance(log_std_range, (tuple, list)) and len(log_std_range) == 2

        super().__init__(**kwargs)

        self.shape = shape
        self.num_actions = np.prod(shape)
        self.std_range = log_std_range

        self.mean = Dense(units=self.num_actions, **kwargs)
        self.log_std = Dense(units=self.num_actions, activation=self._clip_activation, **kwargs)
        self.reshape = Reshape(target_shape=self.shape)

    @tf.function
    def forward_params(self, inputs) -> Tuple[tf.Tensor, tf.Tensor]:
        mean = self.mean(inputs)
        # TODO: consider softplus, or elu + 1
        std = tf.exp(self.log_std(inputs))

        return self.reshape(mean), self.reshape(std)

    def make_distribution_fn(self, params) -> tfd.Normal:
        assert isinstance(params, (list, tuple))
        assert len(params) == 2

        return tfd.Normal(loc=params[0], scale=params[1])

    @tf.function
    def _clip_activation(self, x):
        return tf.minimum(tf.maximum(x, self.std_range[0]), self.std_range[1])


# # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/distributions.py#L195
# class TanhGaussian(Gaussian):
#     """Squashed Gaussian distribution which uses tanh to constrain samples to be in [-1, 1]"""
#
#     class WrappedDistribution(tfd.TransformedDistribution):
#         CLIP_MIN = -1.0 + 1e-5  # utils.TF_EPS
#         CLIP_MAX = +1.0 - 1e-5  # utils.TF_EPS
#
#         def log_prob(self, x, **kwargs):
#             # clip `x` to avoid inf
#             clipped_x = tf.clip_by_value(x, self.CLIP_MIN, self.CLIP_MAX)
#             return super().log_prob(clipped_x)
#
#         def _entropy(self, **kwargs):
#             return None
#
#         def _mean(self, **kwargs):
#             return tf.nn.tanh(self.distribution._mean(**kwargs))
#
#         def _stddev(self, **kwargs):
#             return self.distribution._stddev(**kwargs)
#
#     def make_distribution_fn(self, params) -> tfd.TransformedDistribution:
#         normal = super().make_distribution_fn(params)
#
#         return TanhGaussian.WrappedDistribution(distribution=normal, bijector=tfp.bijectors.Tanh(),
#                                                 name='TanhGaussian')
#
#
# class SquashedGaussian(Gaussian):
#     class TanhNormal(tfd.Normal):
#
#         def _sample_n(self, *args, **kwargs):
#             x = super()._sample_n(*args, **kwargs)
#             return tf.nn.tanh(x)
#
#         def _log_prob(self, x):
#             # th.log(1.0 - th.tanh(x) ** 2 + self.epsilon)
#             log_prob = super()._log_prob(self.tanh_inverse(x))
#             log_prob -= tf.reduce_sum(tf.math.log(1.0 - x**2 + 1e-6), axis=1, keepdims=True)
#             return log_prob
#
#         def _mode(self, **kwargs):
#             mode = super()._mode(**kwargs)
#             return tf.nn.tanh(mode)
#
#         def _entropy(self, **kwargs):
#             return None
#
#         def tanh_inverse(self, x):
#             # atanh
#             x = tf.clip_by_value(x, clip_value_min=-1.0 + 1e-6, clip_value_max=1.0 - 1e-6)
#             return 0.5 * (tf.math.log1p(x) - tf.math.log1p(-x))
#
#     def make_distribution_fn(self, params) -> tfd.Normal:
#         return SquashedGaussian.TanhNormal(loc=params[0], scale=params[1])


class Beta(DistributionLayer):
    """The entropy of Beta (with a, b > 1) is negative in general, and maximum at zero whe a = b = 1.
       (where the Beta equals the uniform distribution). Its minimum is at -infinity.
        - Source: https://en.wikipedia.org/wiki/Beta_distribution#Quantities_of_information_(entropy)
    """

    def __init__(self, shape: tuple, **kwargs):
        assert isinstance(shape, (tuple, list))
        super().__init__(**kwargs)

        self.shape = shape
        self.num_actions = np.prod(shape)

        self.alpha = Dense(units=self.num_actions, activation=self._softplus_one, **kwargs)
        self.beta = Dense(units=self.num_actions, activation=self._softplus_one, **kwargs)
        self.reshape = Reshape(target_shape=self.shape)

    @tf.function
    def forward_params(self, inputs) -> Tuple[tf.Tensor, tf.Tensor]:
        alpha = self.alpha(inputs)
        beta = self.beta(inputs)

        return self.reshape(alpha), self.reshape(beta)

    def make_distribution_fn(self, params) -> tfd.Normal:
        assert isinstance(params, (list, tuple))
        assert len(params) == 2

        # TODO: https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Beta; see WARNING
        return tfd.Beta(concentration1=params[0], concentration0=params[1])

    @tf.function
    def _softplus_one(self, x):
        return tf.nn.softplus(x) + 1.0
