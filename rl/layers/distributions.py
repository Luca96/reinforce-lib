"""Layers for Probability distributions over action spaces"""

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import *

from typing import Dict, Callable, Union, List

from rl import utils


# TODO: replace nan, inf in prob & log_prob
# TODO: put @tf.function to all dist's methods?
class Distribution(Layer):
    """Abstract probability distribution layer that wraps a `tfp.distribution.Distribution` instance"""

    # TODO: "min_log_prob" change -1 to -10
    def __init__(self, independent=True, min_log_prob: Union[str, int, float] = -1, **kwargs):
        super().__init__(**kwargs)

        self.is_independent = bool(independent)
        self.params_shape: tuple = None

        # value at which log-probs are clipped
        if min_log_prob == 'uniform':
            self.min_log_prob = tf.math.log(1.0 / (self.num_actions * self.num_classes))

        elif isinstance(min_log_prob, (int, float)):
            self.min_log_prob = tf.constant(min_log_prob, dtype=tf.float32)
        else:
            raise ValueError(f'"min_log_prob" should be str, int or float not {type(min_log_prob)}.')

    # TODO: use Bernoulli instead of Categorical when there are two (0/1 - bool) actions.
    @staticmethod
    def get(action_space: gym.Space, use_beta=True, **kwargs) -> 'Distribution':
        if isinstance(action_space, gym.spaces.Dict):
            # "action_" is the prefix for complex action spaces
            distributions = {f'action_{k}': Distribution.get(space) for k, space in action_space.spaces.items()}
            return CompoundDistribution(distributions, **kwargs)

        if isinstance(action_space, gym.spaces.Discrete):
            return Categorical(num_actions=1, num_classes=action_space.n, **kwargs)

        # TODO: MultiDiscrete -> Multinomial?

        if isinstance(action_space, gym.spaces.Box):
            # num_actions = action_space.shape[0]
            num_actions = int(np.prod(action_space.shape))

            if action_space.is_bounded():
                if use_beta:
                    return Beta(num_actions, **kwargs)
                else:
                    return TruncatedGaussian(num_actions, low=action_space.low, high=action_space.high, **kwargs)

            return Gaussian(num_actions, **kwargs)

        raise ValueError(f'Unsupported action space type: {type(action_space)}.')

    def deterministic(self, inputs, **kwargs):
        raise NotImplementedError

    def make_distribution_fn(self, params: Union[tf.Tensor, List[tf.Tensor]]) -> tfp.distributions.Distribution:
        base = self.get_base_distribution(params)

        if isinstance(params, (list, tuple)):
            self.params_shape = params[0].shape
        else:
            self.params_shape = params.shape

        if self.num_actions == 1:
            return base

        # num_actions > 1
        if self.is_independent:
            # independent Distribution as single distribution
            return tfp.distributions.Independent(distribution=base, reinterpreted_batch_ndims=1)

        # iid Distribution as single distribution
        return tfp.distributions.Sample(distribution=base, sample_shape=self.num_actions)

    def get_base_distribution(self, params) -> tfp.distributions.Distribution:
        raise NotImplementedError

    def get_built_distribution(self) -> tfp.distributions.Distribution:
        """Returns the most recently instantiated tfp.Distribution"""
        raise NotImplementedError

    def cdf(self, value, **kwargs):
        cdf = self.get_built_distribution().cdf(value, **kwargs)
        return self._reshape(cdf)

    def covariance(self, **kwargs):
        return self.get_built_distribution().covariance(**kwargs)

    def cross_entropy(self, other: 'Distribution', **kwargs):
        cross_ent = self.get_built_distribution().cross_entropy(other=other.get_built_distribution(), **kwargs)
        return utils.tf_flatten(cross_ent)

    def entropy(self, **kwargs):
        ent = self.get_built_distribution().entropy(*kwargs)
        return self._reshape(ent)

    def kl_divergence(self, other: 'Distribution', **kwargs):
        kl = self.get_built_distribution().kl_divergence(other=other.get_built_distribution(), **kwargs)
        return utils.tf_flatten(kl)

    def log_cdf(self, value, **kwargs):
        log_cdf = self.get_built_distribution().log_cdf(value, **kwargs)
        return self._reshape(log_cdf)

    def log_prob(self, value, **kwargs):
        log_prob = self.get_built_distribution().log_prob(value, **kwargs)
        log_prob = tf.clip_by_value(log_prob, clip_value_min=self.min_log_prob, clip_value_max=1.0)

        return self._reshape(log_prob)

    def log_survival_function(self, value, **kwargs):
        log_sf = self.get_built_distribution().log_survival_function(value, **kwargs)
        return self._reshape(log_sf)

    def mean(self, **kwargs):
        mean = self.get_built_distribution().mean(**kwargs)
        return self._reshape(mean, single_dim=False)

    def mode(self, **kwargs):
        mode = self.get_built_distribution().mode(**kwargs)
        return self._reshape(mode, single_dim=False)

    def prob(self, value, **kwargs):
        prob = self.get_built_distribution().prob(value, **kwargs)
        # prob = tf.clip_by_value(prob, clip_value_min=0.0, clip_value_max=1.0)

        return self._reshape(prob)

    def quantile(self, value, **kwargs):
        quantile = self.get_built_distribution().quantile(value, **kwargs)
        return utils.tf_flatten(quantile)

    def sample(self, shape, seed=utils.GLOBAL_SEED, **kwargs):
        return self.get_built_distribution().sample(sample_shape=shape, seed=seed, **kwargs)

    def stddev(self, **kwargs):
        std = self.get_built_distribution().stddev(**kwargs)
        return self._reshape(std, single_dim=False)

    def survival_function(self, value, **kwargs):
        sf = self.get_built_distribution().survival_function(value, **kwargs)
        return self._reshape(sf)

    def variance(self, **kwargs):
        var = self.get_built_distribution().variance(**kwargs)
        return self._reshape(var, single_dim=False)

    @tf.function
    def _reshape(self, x: tf.Tensor, single_dim=True) -> tf.Tensor:
        batch_size = self.params_shape[0]
        dim_size = 1 if single_dim else self.params_shape[-1]

        return tf.reshape(x, shape=(batch_size, dim_size))


class ConcreteDistribution(Distribution):

    def __init__(self, *args, weight_scaling=1.0, **kwargs):
        assert weight_scaling >= 0.0
        super().__init__(*args, **kwargs)

        self.distribution = tfp.layers.DistributionLambda(make_distribution_fn=self.make_distribution_fn)
        self.weight_scaling = tf.constant(weight_scaling, dtype=tf.float32)

    def build(self, input_shape):
        super().build(input_shape)

        if self.weight_scaling == 1.0:
            self._scale_weights()

    def get_built_distribution(self) -> tfp.distributions.Distribution:
        distribution = self.distribution._most_recently_built_distribution

        assert distribution is not None, 'A Distribution instance is build on first inference.'
        return distribution

    def _scale_weights(self):
        raise NotImplementedError


# TODO: multinomial for MultiDiscrete and/or num_actions > 1, OR Independent/Sample categorical?
# TODO: remove `num_actions`
class Categorical(ConcreteDistribution):
    """A layer that wraps a Categorical distribution for discrete actions"""
    def __init__(self, num_actions: int, num_classes: int, name=None, weight_scaling=1.0, **kwargs):
        assert isinstance(num_actions, (int, float))
        assert isinstance(num_classes, (int, float))
        assert num_actions == 1
        assert num_classes >= 1

        # if num_classes=1, the categorical will be deterministic in practice
        super().__init__(name=name, weight_scaling=weight_scaling, trainable=num_classes != 1)

        self.num_actions = int(num_actions)
        self.num_classes = int(num_classes)
        self.shape = (self.num_actions, self.num_classes)

        self.logits = Dense(units=self.num_actions * self.num_classes, activation='linear', **kwargs)
        self.reshape = Reshape(target_shape=self.shape)

    def call(self, inputs, actions=None, **kwargs):
        logits = self.logits(inputs)
        logits = self.reshape(logits)
        categorical: tfp.distributions.Distribution = self.distribution(logits)

        return tf.identity(categorical)

    def deterministic(self, inputs, **kwargs):
        logits = self.logits(inputs)
        logits = self.reshape(logits)
        return tf.argmax(logits, axis=-1)

    def get_base_distribution(self, params) -> tfp.distributions.Distribution:
        return tfp.distributions.Categorical(logits=params)

    def mean(self, **kwargs):
        return tf.zeros(shape=(self.params_shape[0], 1), dtype=tf.float32)

    def stddev(self, **kwargs):
        return tf.zeros(shape=(self.params_shape[0], 1), dtype=tf.float32)

    def _scale_weights(self):
        weights = [w * self.weight_scaling for w in self.logits.get_weights()]
        self.logits.set_weights(weights)


class ContinuousDistribution(ConcreteDistribution):

    def __init__(self, num_actions: int, min_std=1e-2, eps=1e-3, weight_scaling=1.0, name=None, **kwargs):
        assert isinstance(num_actions, (int, float)) >= 1
        super().__init__(name=name, weight_scaling=weight_scaling, **kwargs)

        self.num_actions = int(num_actions)
        self.eps = tf.constant(eps, dtype=tf.float32)
        self.min_std = tf.constant(min_std, dtype=tf.float32)

    def log_prob(self, value, **kwargs):
        return super().log_prob(value=self._round_actions(value), **kwargs)

    def prob(self, value, **kwargs):
        return super().prob(value=self._round_actions(value), **kwargs)

    @tf.function
    def _round_actions(self, actions):
        # round samples (actions) before computing density:
        # https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Beta
        return tf.clip_by_value(actions, self.eps, 1.0 - self.eps)


class Beta(ContinuousDistribution):

    def __init__(self, num_actions: int, min_std=1e-2, eps=1e-3, unimodal=False, independent=True, name=None,
                 weight_scaling=1.0, min_log_prob=-1, **kwargs):
        super().__init__(num_actions, min_std, eps, weight_scaling=weight_scaling, independent=independent, name=name,
                         min_log_prob=min_log_prob)

        if unimodal:
            self.min_std += 1.0

        units = self.num_actions if self.is_independent else 1  # iid

        self.alpha = Dense(units, activation=utils.softplus(self.min_std), **kwargs)
        self.beta = Dense(units, activation=utils.softplus(self.min_std), **kwargs)

    def call(self, inputs, actions=None, **kwargs):
        alpha = self.alpha(inputs)
        beta = self.beta(inputs)

        # TODO: return distribution directly?
        distribution = self.distribution([alpha, beta])
        return tf.identity(distribution)

    # TODO: check this hack!
    def entropy(self, **kwargs):
        return tf.math.abs(super().entropy(**kwargs))

    def get_base_distribution(self, params: list) -> tfp.distributions.Distribution:
        assert len(params) == 2
        return tfp.distributions.Beta(concentration0=params[0], concentration1=params[1])

    def _scale_weights(self):
        w_alpha = [w * self.weight_scaling for w in self.alpha.get_weights()]
        w_beta = [w * self.weight_scaling for w in self.beta.get_weights()]

        self.alpha.set_weights(weights=w_alpha)
        self.beta.set_weights(weights=w_beta)


# TODO: handle iid and independence
# TODO: bound actions using tanh? account for extra term in log-prob
class Gaussian(ContinuousDistribution):

    def __init__(self, num_actions: int, min_std=1e-2, eps=1e-3, weight_scaling=1.0, min_log_prob=-1, name=None,
                 **kwargs):
        super().__init__(num_actions, min_std, eps, weight_scaling=weight_scaling, name=name, min_log_prob=min_log_prob)

        self.mu = Dense(units=self.num_actions, activation='linear', **kwargs)
        self.sigma = Dense(units=self.num_actions, activation=utils.softplus(self.min_std), **kwargs)

    def call(self, inputs, actions=None, **kwargs):
        mu = self.mu(inputs)
        sigma = self.sigma(inputs)

        distribution = self.distribution([mu, sigma])
        return tf.identity(distribution)

    def deterministic(self, inputs, **kwargs):
        return self.mu(inputs)

    def get_base_distribution(self, params: list) -> tfp.distributions.Distribution:
        assert len(params) == 2
        return tfp.distributions.Normal(loc=params[0], scale=params[1])

    def _scale_weights(self):
        w_mean = [w * self.weight_scaling for w in self.mu.get_weights()]
        w_std = [w * self.weight_scaling for w in self.sigma.get_weights()]

        self.mu.set_weights(weights=w_mean)
        self.sigma.set_weights(weights=w_std)


class TruncatedGaussian(Gaussian):

    def __init__(self, num_actions: int, low, high, min_std=1e-2, eps=1e-3, weight_scaling=1.0, min_log_prob=-1,
                 name=None, **kwargs):
        super().__init__(num_actions, min_std, eps, weight_scaling=weight_scaling, name=name, min_log_prob=min_log_prob)

        self.low = tf.constant(low, dtype=tf.float32)
        self.high = tf.constant(high, dtype=tf.float32)

        self.mu = Dense(units=self.num_actions, activation='linear', **kwargs)
        self.sigma = Dense(units=self.num_actions, activation=utils.softplus(self.min_std), **kwargs)

    def get_base_distribution(self, params: list) -> tfp.distributions.Distribution:
        assert len(params) == 2
        return tfp.distributions.TruncatedNormal(loc=params[0], scale=params[1], low=self.low, high=self.high)


class CompoundDistribution(Distribution):

    # TODO: assume independence among distributions, so entropy will sum (as log_prob), and prob is factorized instead.
    def __init__(self, distributions: Dict[str, ConcreteDistribution], method: Union[str, Callable] = 'sum', name=None):
        assert isinstance(distributions, dict) and len(distributions) > 0
        assert all([isinstance(d, ConcreteDistribution) for d in distributions.values()])  # all dist instances, ...
        assert all([not isinstance(d, CompoundDistribution) for d in distributions.values()])  # ...but not compound

        super().__init__(name=name)

        self.distributions = distributions
        self.aggregation_fn = self._get_aggregation_fn(method)

    def call(self, inputs, actions: dict = None, **kwargs):
        return {k: dist(inputs, **kwargs) for k, dist in self.distributions.items()}

    def cdf(self, value: dict, **kwargs):
        assert isinstance(value, dict)
        cdf = []

        for k, dist in self.distributions.items():
            cdf_ = dist.cdf(value=value[k], **kwargs)
            cdf.append(cdf_[:, None])

        return self.aggregation_fn(tf.concat(cdf, axis=-1))

    def covariance(self, **kwargs):
        cov = [d.covariance(**kwargs)[:, None] for d in self.distributions.values()]
        return self.aggregation_fn(tf.concat(cov, axis=-1))

    def cross_entropy(self, other: 'CompoundDistribution', **kwargs):
        assert self.distributions.keys() == other.distributions.keys()
        cross_ent = []

        for key, dist in self.distributions.items():
            ent = dist.cross_entropy(other=other.distributions[key], **kwargs)
            cross_ent.append(ent)

        return self.aggregation_fn(tf.concat(cross_ent, axis=-1))

    def entropy(self, **kwargs):
        entropy = [d.entropy(**kwargs) for d in self.distributions.values()]
        return self.aggregation_fn(tf.concat(entropy, axis=-1))

    def kl_divergence(self, other: 'CompoundDistribution', **kwargs):
        assert self.distributions.keys() == other.distributions.keys()
        kld = []

        for key, dist in self.distributions.items():
            kl = dist.kl_divergence(other=other.distributions[key], **kwargs)
            kld.append(kl)

        return self.aggregation_fn(tf.concat(kld, axis=-1))

    def log_cdf(self, value: dict, **kwargs):
        assert isinstance(value, dict)
        log_cdf = []

        for k, dist in self.distributions.items():
            log_cdf_ = dist.log_cdf(value=tf.squeeze(value[k]), **kwargs)
            log_cdf.append(log_cdf_)

        return self.aggregation_fn(tf.concat(log_cdf, axis=-1))

    def log_prob(self, value: dict, **kwargs):
        assert isinstance(value, dict)
        log_prob = []

        for k, dist in self.distributions.items():
            log_prob_ = dist.log_prob(value=value[k], **kwargs)
            log_prob.append(log_prob_)

        return self.aggregation_fn(tf.concat(log_prob, axis=-1))

    def log_survival_function(self, value: dict, **kwargs):
        assert isinstance(value, dict)
        log_sf = []

        for k, dist in self.distributions.items():
            log_sf_ = dist.log_survival_function(value=tf.squeeze(value[k]), **kwargs)
            log_sf.append(log_sf_)

        return self.aggregation_fn(tf.concat(log_sf, axis=-1))

    def mean(self, **kwargs):
        mean = [d.mean(**kwargs) for d in self.distributions.values()]
        return self.aggregation_fn(tf.concat(mean, axis=-1))

    def mode(self, **kwargs):
        mode = [d.mode(**kwargs) for d in self.distributions.values()]
        return self.aggregation_fn(tf.concat(mode, axis=-1))

    # def prob(self, value: dict, **kwargs):
    #     assert isinstance(value, dict)
    #     prob = []
    #
    #     for k, dist in self.distributions.items():
    #         p = dist.prob(value=tf.squeeze(value[k]), **kwargs)
    #         prob.append(p)
    #
    #     return self.aggregation_fn(tf.concat(prob, axis=-1))

    def prob(self, value: dict, **kwargs):
        assert isinstance(value, dict)
        prob = 1.0

        for k, dist in self.distributions.items():
            prob *= dist.prob(value=tf.squeeze(value[k]), **kwargs)

        return prob

    def quantile(self, value: dict, **kwargs):
        assert isinstance(value, dict)
        quantile = []

        for k, dist in self.distributions.items():
            q = dist.quantile(value=tf.squeeze(value[k]), **kwargs)
            quantile.append(q)

        return self.aggregation_fn(tf.concat(quantile, axis=-1))

    def sample(self, shape, seed=utils.GLOBAL_SEED, **kwargs) -> dict:
        return {k: d.sample(sample_shape=shape, seed=seed, **kwargs) for k, d in self.distributions.items()}

    def stddev(self, **kwargs):
        std = [d.stddev(**kwargs) for d in self.distributions.values()]
        return self.aggregation_fn(tf.concat(std, axis=-1))

    def survival_function(self, value: dict, **kwargs):
        assert isinstance(value, dict)
        sf = []

        for k, dist in self.distributions.items():
            sf_ = dist.survival_function(value=tf.squeeze(value[k]), **kwargs)
            sf.append(sf_)

        return self.aggregation_fn(tf.concat(sf, axis=-1))

    def variance(self, **kwargs):
        var = [d.variance(**kwargs) for d in self.distributions.values()]
        return self.aggregation_fn(tf.concat(var, axis=-1))

    # TODO: wrap into @tf.function
    # TODO: add weighted average?
    @staticmethod
    def _get_aggregation_fn(method):
        if callable(method):
            return method

        assert isinstance(method, str)
        method = method.lower()

        if method == 'add':
            return lambda x: tf.reduce_sum(x, axis=-1, keepdims=True)

        if method in ['mean', 'avg', 'average']:
            return lambda x: tf.reduce_mean(x, axis=-1, keepdims=True)

        if method in ['max', 'maximum']:
            return lambda x: tf.reduce_max(x, axis=-1, keepdims=True)

        if method in ['min', 'minimum']:
            return lambda x: tf.reduce_min(x, axis=-1, keepdims=True)

        raise ValueError(f'Unknown aggregation method "{method}"')

    def get_built_distribution(self):
        raise NotImplementedError
