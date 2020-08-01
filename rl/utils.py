import os
import gym
import math
import numpy as np
import scipy.signal
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import random

from typing import Union, List, Dict, Tuple
from datetime import datetime

from gym import spaces


# -------------------------------------------------------------------------------------------------
# -- Constants
# -------------------------------------------------------------------------------------------------

EPSILON = tf.constant(np.finfo(np.float32).eps, dtype=tf.float32)

OPTIMIZERS = dict(adadelta=tf.keras.optimizers.Adadelta,
                  adagrad=tf.keras.optimizers.Adagrad,
                  adam=tf.keras.optimizers.Adam,
                  adamax=tf.keras.optimizers.Adamax,
                  ftrl=tf.keras.optimizers.Ftrl,
                  nadam=tf.keras.optimizers.Nadam,
                  rmsprop=tf.keras.optimizers.RMSprop,
                  sgd=tf.keras.optimizers.SGD)


def get_optimizer_by_name(name: str, *args, **kwargs) -> tf.keras.optimizers.Optimizer:
    optimizer_class = OPTIMIZERS.get(name.lower(), None)

    if optimizer_class is None:
        raise ValueError(f'Cannot find optimizer {name}. Select one of {OPTIMIZERS.keys()}.')

    print(f'Optimizer: {name}.')
    return optimizer_class(*args, **kwargs)


# -------------------------------------------------------------------------------------------------
# -- Misc
# -------------------------------------------------------------------------------------------------

def np_normalize(x, epsilon=np.finfo(np.float32).eps):
    return (x - np.mean(x)) / (np.std(x) + epsilon)


def discount_cumsum(x, discount: float):
    """Source: https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/ppo/core.py#L45"""
    return scipy.signal.lfilter([1.0], [1.0, float(-discount)], x[::-1], axis=0)[::-1]


def gae(rewards, values, gamma: float, lambda_: float, normalize=False):
    rewards = tf.expand_dims(rewards, axis=-1)
    deltas = rewards[:-1] + tf.math.multiply(values[1:], gamma) - values[:-1]
    advantages = discount_cumsum(deltas, discount=gamma * lambda_)

    if normalize:
        advantages = tf_normalize(advantages)

    return advantages


def rewards_to_go(rewards, discount: float, normalize=False):
    returns = discount_cumsum(rewards, discount=discount)[:-1]

    if normalize:
        returns = np_normalize(returns)

    return returns


def is_image(x) -> bool:
    """Checks whether some input [x] has a shape of the form (H, W, C)"""
    return len(x.shape) == 3


def is_vector(x) -> bool:
    """Checks whether some input [x] has a shape of the form (N, D) or (D,)"""
    return 1 <= len(x.shape) <= 2


def depth_concat(*arrays):
    return np.concatenate(*arrays, axis=-1)


def plot_images(images: list):
    """Plots a list of images, arranging them in a rectangular fashion"""
    num_plots = len(images)
    rows = round(math.sqrt(num_plots))
    cols = math.ceil(math.sqrt(num_plots))

    for k, img in enumerate(images):
        plt.subplot(rows, cols, k + 1)
        plt.axis('off')
        plt.imshow(img)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


# -------------------------------------------------------------------------------------------------
# -- Gym utils
# -------------------------------------------------------------------------------------------------

def print_info(gym_env):
    if isinstance(gym_env, str):
        gym_env = gym.make(gym_env)

    obs_space = gym_env.observation_space
    act_space = gym_env.action_space

    # Observation space:
    if isinstance(obs_space, gym.spaces.Box):
        print(f'Observation space: {obs_space}, shape: {obs_space.shape}, bounds: {obs_space.low}, {obs_space.high}')
    else:
        print(f'Observation space: {obs_space}, n: {obs_space.n}')

    # Action space:
    if isinstance(act_space, gym.spaces.Box):
        print(f'Action space: {act_space}, shape: {act_space.shape}, bounds: {act_space.low}, {act_space.high}')
    else:
        print(f'Action space: {act_space}, n: {act_space.n}')

    print('Reward range:', gym_env.reward_range)
    print('Metadata:', gym_env.metadata)


def space_to_flat_spec(space: gym.Space, name: str) -> Dict[str, tuple]:
    """From a gym.Space object returns a flat dictionary str -> tuple.
       Naming convention:
         - If space is Box or Discrete, it returns 'dict(name=shape)'
         - If space is Dict (not nested), it returns 'dict(name_x=shape_x, name_y=shape_y)'
            considering 'x' and 'y' be component of space.
         - With further nesting, dict keys' names got created using the above two rules.
           In this way each key (name) uniquely identifies a (sub-)component of the space.
           Example:
              Dict(a=x, b=Dict(c=y, d=z)) -> dict(a=x, b_c=y, b_d=z)
    """
    spec = dict()

    if isinstance(space, spaces.Discrete):
        spec[name] = (space.n,)

    elif isinstance(space, spaces.MultiDiscrete):
        spec[name] = space.nvec.shape

    elif isinstance(space, spaces.Box):
        spec[name] = space.shape

    elif isinstance(space, spaces.Dict):
        for key, value in space.spaces.items():
            space_name = f'{name}_{key}'
            result = space_to_flat_spec(space=value, name=space_name)

            if isinstance(result, dict):
                for k, v in result.items():
                    spec[k] = v
            else:
                spec[space_name] = result
    else:
        raise ValueError('space must be one of Box, Discrete, MultiDiscrete, or Dict')

    return spec


def space_to_spec(space: gym.Space) -> Union[tuple, Dict[str, Union[tuple, dict]]]:
    """From a gym.Space object returns its shape-specification, i.e.
         - tuple: if space is Box or Discrete
         - dict[str, tuple or dict]: if space is spaces.Dict
    """
    if isinstance(space, spaces.Box):
        return space.shape

    if isinstance(space, spaces.Discrete):
        return space.n,  # -> tuple (space.n,)

    if isinstance(space, spaces.MultiDiscrete):
        return space.nvec.shape

    assert isinstance(space, spaces.Dict)

    spec = dict()
    for name, space in space.spaces.items():
        # use recursion to handle arbitrary nested Dicts
        spec[name] = space_to_spec(space)

    return spec


# -------------------------------------------------------------------------------------------------
# -- TF utils
# -------------------------------------------------------------------------------------------------

# TODO: @tf.function
def to_tensor(x, expand_axis=0):
    if isinstance(x, dict):
        for k, v in x.items():
            v = tf.cast(v, dtype=tf.float32)
            x[k] = tf.expand_dims(tf.convert_to_tensor(v), axis=expand_axis)
    else:
        x = tf.cast(x, dtype=tf.float32)
        x = tf.convert_to_tensor(x)
        x = tf.expand_dims(x, axis=expand_axis)
    return x


# TODO: @tf.function
def tf_normalize(x):
    """Normalizes some tensor x to 0-mean 1-stddev"""
    return (x - tf.math.reduce_mean(x)) / tf.math.reduce_std(x)


def data_to_batches(tensors: Union[List, Tuple], batch_size: int, shuffle_batches=False, seed=None,
                    drop_remainder=False, map_fn=None, prefetch_size=2, num_shards=1, skip=0):
    """Transform some tensors data into a dataset of mini-batches"""
    dataset = tf.data.Dataset.from_tensor_slices(tensors).skip(count=skip)

    if num_shards > 1:
        # "observation skip trick" with tf.data.Dataset.shard()
        ds = dataset.shard(num_shards, index=0)

        for shard_index in range(1, num_shards):
            shard = dataset.shard(num_shards, index=shard_index)
            ds = ds.concatenate(shard)

        dataset = ds

    if map_fn is not None:
        # 'map_fn' is mainly used for 'data augmentation'
        dataset = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE,
                              deterministic=True)

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    if shuffle_batches:
        dataset = dataset.shuffle(buffer_size=batch_size, seed=seed)

    return dataset.prefetch(buffer_size=prefetch_size)


# TODO: @tf.function
def tf_to_scalar_shape(tensor):
    return tf.reshape(tensor, shape=[])


def assert_shapes(a, b):
    assert tf.shape(a) == tf.shape(b)


def tf_01_scaling(x):
    x -= tf.reduce_min(x)
    x /= tf.reduce_max(x)
    return x


@tf.function
def softplus_one(x):
    return 1.0 + tf.nn.softplus(x)


@tf.function
def batch_norm_relu6(layer: tf.keras.layers.Layer):
    """BatchNormalization + ReLU6, use as activation function"""
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.nn.relu6(layer)
    return layer


@tf.function
def kl_divergence(log_a, log_b):
    """Kullback-Leibler divergence
        - Source: https://www.tensorflow.org/api_docs/python/tf/keras/losses/KLD
    """
    return log_a * (log_a - log_b)


@tf.function
def tf_entropy(prob, log_prob):
    return -tf.reduce_sum(prob * log_prob)


def to_int(tensor):
    """Casts the given tensor to tf.int32 datatype"""
    return tf.cast(tensor, dtype=tf.int32)


def to_float(tensor):
    """Casts the given tensor to tf.float32 datatype"""
    return tf.cast(tensor, dtype=tf.float32)


# -------------------------------------------------------------------------------------------------
# -- File utils
# -------------------------------------------------------------------------------------------------

def makedir(*args: str) -> str:
    """Creates a directory"""
    path = os.path.join(*args)
    os.makedirs(path, exist_ok=True)
    return path


def file_names(dir_path: str, sort=True) -> list:
    files = filter(lambda f: os.path.isfile(os.path.join(dir_path, f)) and f.startswith('trace-')
                   and f.endswith('.npz'), os.listdir(dir_path))
    if sort:
        files = sorted(files)

    return list(files)


def load_traces(traces_dir: str, shuffle=False):
    if shuffle:
        trace_names = file_names(traces_dir, sort=False)
        random.shuffle(trace_names)
    else:
        trace_names = file_names(traces_dir, sort=True)

    for name in trace_names:
        yield np.load(file=os.path.join(traces_dir, name))


def unpack_trace(trace: dict) -> tuple:
    """Reads a trace (i.e. a dict-like object created by np.load()) and unpacks it as a tuple
       (state, action, reward, done).
    """
    trace_keys = trace.keys()
    trace = {k: trace[k] for k in trace_keys}  # copy

    for name in ['state', 'action']:
        # check if state/action space is simple (array, i.e sum == 1) or complex (dict of arrays)
        if sum(k.startswith(name) for k in trace_keys) == 1:
            continue

        # select keys of the form 'state_xxx', then build a dict(state_x=trace['state_x'])
        keys = filter(lambda k: k.startswith(name + '_'), trace_keys)
        trace[name] = {k: trace[k] for k in keys}

    return trace['state'], trace['action'], trace['reward'], trace['done']


# -------------------------------------------------------------------------------------------------
# -- Statistics utils
# -------------------------------------------------------------------------------------------------

class Statistics:
    def __init__(self, mode='summary', name=None, summary_dir='logs'):
        self.stats = dict()

        if mode == 'summary':
            self.should_log = True
            self.use_summary = True

        elif mode == 'log':
            self.should_log = True
            self.use_summary = False
        else:
            self.should_log = False
            self.use_summary = False

        if self.use_summary:
            self.summary_dir = os.path.join(summary_dir, name, datetime.now().strftime("%Y%m%d-%H%M%S"))
            self.tf_summary_writer = tf.summary.create_file_writer(self.summary_dir)

    def log(self, **kwargs):
        if not self.should_log:
            return

        for key, value in kwargs.items():
            if key not in self.stats:
                self.stats[key] = dict(step=0, list=[])

            if tf.is_tensor(value):
                if np.prod(value.shape) > 1:
                    self.stats[key]['list'].extend(value)
                else:
                    self.stats[key]['list'].append(value)

            elif hasattr(value, '__iter__'):
                self.stats[key]['list'].extend(value)
            else:
                self.stats[key]['list'].append(value)

    def write_summaries(self):
        if not self.use_summary:
            return

        with self.tf_summary_writer.as_default():
            for summary_name, data in self.stats.items():
                step = data['step']
                values = data['list']

                for i, value in enumerate(values):
                    # TODO: 'np.mean' is a temporary fix...
                    tf.summary.scalar(name=summary_name, data=np.mean(value), step=step + i)

                # clear value_list, update step
                self.stats[summary_name]['step'] += len(values)
                self.stats[summary_name]['list'].clear()

    def plot(self, colormap='Set3'):  # Pastel1, Set3, tab20b, tab20c
        """Colormaps: https://matplotlib.org/tutorials/colors/colormaps.html"""
        num_plots = len(self.stats.keys())
        cmap = plt.get_cmap(name=colormap)
        rows = round(math.sqrt(num_plots))
        cols = math.ceil(math.sqrt(num_plots))

        for k, (key, value) in enumerate(self.stats.items()):
            plt.subplot(rows, cols, k + 1)
            plt.plot(value, color=cmap(k + 1))
            plt.title(key)

        plt.show()


class IncrementalStatistics:
    """Compute mean, variance, and standard deviation incrementally."""
    def __init__(self):
        self.mean = 0.0
        self.variance = 0.0
        self.std = 0.0
        self.count = 0

    def update(self, x):
        old_mean = self.mean
        new_mean = tf.reduce_mean(x)
        m = self.count
        n = tf.shape(x)[0]
        c1 = m / (m + n)
        c2 = n / (m + n)
        c3 = (m * n) / (m + n) ** 2

        self.mean = c1 * old_mean + c2 * new_mean
        self.variance = c1 * self.variance + c2 * tf.math.reduce_variance(x) + c3 * (old_mean - new_mean) ** 2
        self.std = tf.sqrt(self.variance)
        self.count += n

    def set(self, mean: float, variance: float, std: float, count: int):
        self.mean = mean
        self.variance = variance
        self.std = std
        self.count = count

    def as_dict(self) -> dict:
        return dict(mean=np.float(self.mean), variance=np.float(self.variance),
                    std=np.float(self.std), count=np.int(self.count))


# -------------------------------------------------------------------------------------------------
# -- Distributions utils
# -------------------------------------------------------------------------------------------------

class MixtureDistribution(tfp.distributions.Mixture):

    def entropy(self, name='entropy', **kwargs):
        return super().entropy_lower_bound()

    def kl_divergence(self, other, name='kl_divergence'):
        """Roughly approximated KL-divergence"""
        approx_kl = 0.0

        for i, component in enumerate(self.components):
            approx_kl += self.cat.prob(i) * component.kl_divergence(other)

        return approx_kl


def get_mixture_of_categorical(layer: tf.keras.layers.Layer, num_actions: int,
                               num_components: int) -> tfp.layers.DistributionLambda:
    layers = []

    # define the layers that weights the mixture's components
    weights = tf.keras.layers.Dense(units=num_components)(layer)
    weights = tf.expand_dims(weights, axis=0, name='mixture-weights')
    layers.append(weights)

    # create a logits (dense) layer for each component
    for i in range(num_components):
        logits = tf.keras.layers.Dense(units=num_actions, activation='linear')(layer)
        logits = tf.expand_dims(logits, axis=0, name=f'logits-{i + 1}')
        layers.append(logits)

    # make the distribution lambda layer that wraps the mixture
    return tfp.layers.DistributionLambda(
        make_distribution_fn=lambda t: MixtureDistribution(
            cat=tfp.distributions.Categorical(logits=t[0]),
            components=[tfp.distributions.Categorical(logits=t[j + 1]) for j in range(num_components)]
        )
    )(layers)


def get_mixture_of_beta(layer: tf.keras.layers.Layer, num_actions: int,
                        num_components: int) -> tfp.layers.DistributionLambda:
    layers = []

    # define the layers that weights the mixture's components
    weights = tf.keras.layers.Dense(units=num_components, name='mixture-weights')(layer)
    layers.append(weights)

    # create a dense layer for alpha and beta parameters for each component
    for i in range(num_components):
        alpha = tf.keras.layers.Dense(units=num_actions, activation='softplus')(layer)
        alpha = tf.keras.layers.Add(name=f'alpha-{i}')([alpha, tf.ones_like(alpha)])

        beta = tf.keras.layers.Dense(units=num_actions, activation='softplus')(layer)
        beta = tf.keras.layers.Add(name=f'beta-{i}')([beta, tf.ones_like(beta)])

        layers.append([alpha, beta])

    # make the distribution lambda layer that wraps the mixture
    return tfp.layers.DistributionLambda(
        make_distribution_fn=lambda t: MixtureDistribution(
            cat=tfp.distributions.Categorical(logits=t[0]),
            components=[tfp.distributions.Beta(t[j + 1][0], t[j + 1][1]) for j in range(num_components)]
        )
    )(layers)


def get_mixture_of_gaussian(layer: tf.keras.layers.Layer, num_actions: int,
                            num_components: int) -> tfp.layers.DistributionLambda:
    layers = []

    # define the layers that weights the mixture's components
    weights = tf.keras.layers.Dense(units=num_components, name='mixture-weights')(layer)
    layers.append(weights)

    # create a dense layer for alpha and beta parameters for each component
    for i in range(num_components):
        mu = tf.keras.layers.Dense(units=num_actions, activation='linear', name=f'mu-{i}')(layer)
        sigma = tf.keras.layers.Dense(units=num_actions, activation='softplus', name=f'sigma-{i}')(layer)

        layers.append([mu, sigma])

    # make the distribution lambda layer that wraps the mixture
    return tfp.layers.DistributionLambda(
        make_distribution_fn=lambda t: MixtureDistribution(
            cat=tfp.distributions.Categorical(logits=t[0]),
            components=[tfp.distributions.MultivariateNormalDiag(
                loc=t[j + 1][0], scale_diag=t[j + 1][1]) for j in range(num_components)]
        )
    )(layers)


def sample_and_scale01(d: tfp.distributions.Distribution):
    # be sure `d` is a  multivariate Gaussian distribution
    assert isinstance(d, tfp.distributions.MultivariateNormalDiag)

    # distribution's support (3-sigma rule)
    min_value = -3.0 * d.stddev()
    max_value = +3.0 * d.stddev()

    # sample and scale it in 0-1 interval:
    sample = tf.clip_by_value(d.sample(), min_value, max_value)

    return (sample - min_value) / (max_value - min_value)
