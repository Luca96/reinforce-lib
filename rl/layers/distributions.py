"""Layers for Probability distributions over action spaces"""

import gym
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import *

from typing import Dict, Callable, Union

from rl import utils


class Distribution(Layer):
    """Probability distribution layer that wraps a `tfp.distribution.Distribution` instance"""

    @staticmethod
    def get(action_space: gym.Space, use_beta=True, **kwargs) -> 'Distribution':
        if isinstance(action_space, gym.spaces.Dict):
            distributions = {k: Distribution.get(space) for k, space in action_space.spaces.items()}
            return CompoundDistribution(distributions, **kwargs)

        if isinstance(action_space, gym.spaces.Discrete):
            return Categorical(num_actions=1, num_classes=action_space.n, **kwargs)

        if isinstance(action_space, gym.spaces.Box):
            num_actions = action_space.shape[0]

            if action_space.is_bounded():
                if use_beta:
                    return Beta(num_actions, **kwargs)
                else:
                    return TruncatedGaussian(num_actions, low=action_space.low, high=action_space.high, **kwargs)

            return Gaussian(num_actions, **kwargs)

        raise ValueError(f'Unsupported action space type: {type(action_space)}.')


class Categorical(Distribution):
    """A layer that wraps a Categorical distribution for discrete actions"""
    def __init__(self, num_actions: int, num_classes: int,  name=None, **kwargs):
        assert isinstance(num_actions, (int, float)) >= 1
        assert isinstance(num_classes, (int, float)) >= 1

        super().__init__(name=name)

        self.num_actions = int(num_actions)
        self.num_classes = int(num_classes)
        self.shape = (self.num_actions, self.num_classes)

        self.logits = Dense(units=self.num_actions * self.num_classes, activation='linear', **kwargs)
        self.reshape = Reshape(target_shape=self.shape)

        self.distribution = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfp.distributions.Categorical(logits=t))

    def call(self, inputs, actions=None, **kwargs):
        logits = self.logits(inputs)
        logits = self.reshape(logits)
        categorical: tfp.distributions.Distribution = self.distribution(logits)

        if tf.is_tensor(actions):
            return categorical.log_prob(actions), categorical.entropy()

        # output, log-probability, mean, and stddev
        mean = tf.constant(0.0, dtype=tf.float32)
        std = tf.constant(0.0, dtype=tf.float32)

        return categorical, categorical.log_prob(categorical), mean, std


class ContinuousDistribution(Distribution):

    def __init__(self, num_actions: int, min_std=1e-2, eps=1e-3, name=None):
        assert isinstance(num_actions, (int, float)) >= 1
        super().__init__(name=name)

        self.num_actions = int(num_actions)
        self.eps = tf.constant(eps, dtype=tf.float32)
        self.min_std = tf.constant(min_std, dtype=tf.float32)

    @tf.function
    def _round_actions(self, actions):
        # round samples (actions) before computing density:
        # https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Beta
        return tf.clip_by_value(actions, self.eps, 1.0 - self.eps)


class Beta(ContinuousDistribution):

    def __init__(self, num_actions: int, min_std=1e-2, eps=1e-3, unimodal=False, name=None, **kwargs):
        super().__init__(num_actions, min_std, eps, name=name)

        if unimodal:
            self.min_std += 1.0

        self.alpha = Dense(units=self.num_actions, activation=utils.softplus(self.min_std), **kwargs)
        self.beta = Dense(units=self.num_actions, activation=utils.softplus(self.min_std), **kwargs)

        self.distribution = tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Beta(t[0], t[1]))

    def call(self, inputs, actions=None, **kwargs):
        alpha = self.alpha(inputs)
        beta = self.beta(inputs)

        distribution: tfp.distributions.Distribution = self.distribution([alpha, beta])
        new_actions = self._round_actions(distribution)

        if tf.is_tensor(actions):
            actions = self._round_actions(actions)
            return distribution.log_prob(actions), distribution.entropy()

        return new_actions, distribution.log_prob(new_actions), distribution.mean(), distribution.stddev()


class Gaussian(ContinuousDistribution):

    def __init__(self, num_actions: int, min_std=1e-2, eps=1e-3, name=None, **kwargs):
        super().__init__(num_actions, min_std, eps, name=name)

        self.mu = Dense(units=self.num_actions, activation='linear', **kwargs)
        self.sigma = Dense(units=self.num_actions, activation=utils.softplus(self.min_std), **kwargs)

        self.distribution = tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Normal(t[0], t[1]))

    def call(self, inputs, actions=None, **kwargs):
        mu = self.mu(inputs)
        sigma = self.sigma(inputs)

        gaussian: tfp.distributions.Distribution = self.distribution([mu, sigma])
        new_actions = self._round_actions(gaussian)

        if tf.is_tensor(actions):
            actions = self._round_actions(actions)
            return gaussian.log_prob(actions), gaussian.entropy()

        return new_actions, gaussian.log_prob(new_actions), gaussian.mean(), gaussian.stddev()


class TruncatedGaussian(Gaussian):

    def __init__(self, num_actions: int, low, high, min_std=1e-2, eps=1e-3, name=None, **kwargs):
        super().__init__(num_actions, min_std, eps, name=name)

        self.low = tf.constant(low, dtype=tf.float32)
        self.high = tf.constant(high, dtype=tf.float32)

        self.mu = Dense(units=self.num_actions, activation='linear', **kwargs)
        self.sigma = Dense(units=self.num_actions, activation=utils.softplus(self.min_std), **kwargs)

        self.distribution = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfp.distributions.TruncatedNormal(t[0], t[1], low=self.low, high=self.high))


class CompoundDistribution(Distribution):

    def __init__(self, distributions: Dict[str, Distribution], method: Union[str, Callable] = 'avg', name=None):
        assert isinstance(distributions, dict) and len(distributions) > 0
        assert all([isinstance(d, Distribution) for d in distributions.values()])  # all distribution instances, ...
        assert all([not isinstance(d, CompoundDistribution) for d in distributions.values()])  # ...but not compound

        super().__init__(name=name)

        self.distributions = distributions
        self.aggregation_fn = self._get_aggregation_fn(method)

    def call(self, inputs, actions: dict = None, **kwargs):
        if isinstance(actions, dict):
            outputs = [self.distributions[key](inputs, actions=action, **kwargs) for key, action in actions.items()]
            outputs = tf.convert_to_tensor(outputs, dtype=tf.float32)

            # then aggregate
            log_prob = self.aggregation_fn(outputs[:, 0])
            entropy = self.aggregation_fn(outputs[:, 1])  # TODO: multiply by `prob` for weighted average?

            # TODO: return dict for debug
            return log_prob, entropy

        actions = dict()
        log_prob = dict()
        mean = dict()
        std = dict()

        for key, distribution in self.distributions.items():
            out = distribution(inputs, actions=None, **kwargs)

            actions[key] = out[0]
            log_prob[key] = out[1]
            mean[key] = out[2]
            std[key] = out[3]

        return actions, log_prob, mean, std

    # TODO: add weighted average?
    @staticmethod
    def _get_aggregation_fn(method):
        if callable(method):
            return method

        assert isinstance(method, str)
        method = method.lower()

        if method == 'add':
            return tf.reduce_sum

        if method in ['mean', 'avg', 'average']:
            return tf.reduce_mean

        if method in ['max', 'maximum']:
            return tf.reduce_max

        if method in ['min', 'minimum']:
            return tf.reduce_min

        raise ValueError(f'Unknown aggregation method "{method}"')
