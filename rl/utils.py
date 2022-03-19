import os
import gym
import math
import numpy as np
import scipy.signal
import tensorflow as tf
import matplotlib.pyplot as plt
import multiprocessing as mp
import random
import time

from typing import Union, List, Dict, Tuple, Optional, Callable, Any
from distutils import dir_util
from datetime import datetime

from gym import spaces

from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from rl.parameters import DynamicParameter


# -------------------------------------------------------------------------------------------------
# -- Constants
# -------------------------------------------------------------------------------------------------

GLOBAL_SEED: int = None

NP_EPS = np.finfo(np.float32).eps
TF_EPS = tf.constant(NP_EPS, dtype=tf.float32)

TF_ZERO = tf.constant(0.0, dtype=tf.float32)
TF_PI = tf.constant(np.pi, dtype=tf.float32)

OPTIMIZERS = dict(adadelta=tf.keras.optimizers.Adadelta,
                  adagrad=tf.keras.optimizers.Adagrad,
                  adam=tf.keras.optimizers.Adam,
                  adamax=tf.keras.optimizers.Adamax,
                  ftrl=tf.keras.optimizers.Ftrl,
                  nadam=tf.keras.optimizers.Nadam,
                  rmsprop=tf.keras.optimizers.RMSprop,
                  sgd=tf.keras.optimizers.SGD)

# Types
DynamicType = Union[float, LearningRateSchedule, DynamicParameter]
TensorDictOrTensor = Union[Dict[str, tf.Tensor], tf.Tensor]


def get_optimizer_by_name(name: str, *args, **kwargs) -> tf.keras.optimizers.Optimizer:
    name = name.lower()
    optimizer_class = OPTIMIZERS.get(name, None)

    if optimizer_class is None:
        raise ValueError(f'Cannot find optimizer {name}. Select one of {OPTIMIZERS.keys()}.')

    if name == 'adam':
        # Source: Google Dopamine
        kwargs.setdefault('epsilon', 0.0003125)  # more numerically stable (maybe..)

    elif name == 'rmsprop':
        # https://github.com/google/dopamine/blob/master/dopamine/agents/dqn/configs/dqn_nature.gin#L22
        kwargs.setdefault('epsilon', 0.00001)

    print(f'Optimizer: {name}.')
    return optimizer_class(*args, **kwargs)


def get_optimizer(optimizer: Union[str, dict], **kwargs) -> tf.keras.optimizers.Optimizer:
    if isinstance(optimizer, dict):
        name = optimizer.pop('name').lower()
        return get_optimizer_by_name(name, **optimizer, **kwargs)

    assert isinstance(optimizer, str)
    return get_optimizer_by_name(name=optimizer, **kwargs)


def get_normalization_layer(name: str, **kwargs) -> tf.keras.layers.Layer:
    if isinstance(name, str):
        name = name.lower()
    else:
        raise ValueError(f'Parameter `name` should be a "string" not "{type(name)}"')

    if name == 'batch':
        return tf.keras.layers.BatchNormalization(**kwargs)

    if name == 'layer':
        return tf.keras.layers.LayerNormalization(**kwargs)


def apply_normalization(layer: tf.keras.layers.Layer, name: str, **kwargs) -> tf.keras.layers.Layer:
    if not isinstance(name, str):
        return layer

    name = name.lower()

    if name == 'batch':
        return tf.keras.layers.BatchNormalization(**kwargs)(layer)

    if name == 'layer':
        return tf.keras.layers.LayerNormalization(**kwargs)(layer)


def get_normalization_fn(arg: Union[str, Callable] = None, **kwargs):
    if callable(arg):
        return arg

    arg = str(arg or 'identity').lower()
    assert arg in ['identity', 'standard', 'sign', 'min_max', 'minmax', 'magnitude']

    if arg == 'identity':
        return lambda x: x

    if arg == 'standard':
        return lambda x: tf_normalize(x, **kwargs)

    if arg == 'sign':
        return lambda x: tf_sp_norm(x, **kwargs)

    if arg == 'min_max' or arg == 'minmax':
        return lambda x: tf_minmax_norm(x, **kwargs)

    if arg == 'magnitude':
        return tf_magnitude_norm


def pooling_2d(layer: tf.keras.layers.Layer, args: Union[str, dict]) -> tf.keras.layers.Layer:
    if args is None:
        return layer

    if isinstance(args, dict):
        which = args.pop('which', args.pop('name', args.pop('type', 'max')))
    else:
        which = args
        args = {}

    assert isinstance(which, str)
    which = which.lower()

    if which == 'max':
        return tf.keras.layers.MaxPooling2D(**args)(layer)

    if which == 'avg':
        return tf.keras.layers.AveragePooling2D(**args)(layer)

    raise ValueError(f'Unknown Pooling layer with name "{which}", use "max" or "avg" only.')


def global_pool2d_or_flatten(arg: Union[str, bool]) -> tf.keras.layers.Layer:
    if (arg is None) or (arg is False):
        return tf.keras.layers.Flatten()

    assert isinstance(arg, str)
    arg = arg.lower()

    if arg == 'max':
        return tf.keras.layers.GlobalMaxPooling2D()

    if arg == 'avg':
        return tf.keras.layers.GlobalAveragePooling2D()

    raise ValueError(f'Unknown GlobalPooling layer with name "{arg}", use "max" or "avg" only.')


# -------------------------------------------------------------------------------------------------
# -- Misc
# -------------------------------------------------------------------------------------------------

def set_random_seed(seed=None):
    """Sets the random seed for tensorflow, numpy, python's random"""
    global GLOBAL_SEED

    if seed is not None:
        seed = int(seed)
        assert 0 <= seed < 2 ** 32

        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        GLOBAL_SEED = seed
        print(f'Global random seed {seed} set.')


def get_random_generator(seed=None) -> np.random.Generator:
    """Returns a numpy's random generator instance"""
    if seed is not None:
        seed = int(seed)
        assert 0 <= seed < 2 ** 32
    else:
        seed = GLOBAL_SEED

    return np.random.default_rng(np.random.MT19937(seed=seed))


def get_seed(seed):
    if seed is None:
        return GLOBAL_SEED

    assert isinstance(seed, (float, int))
    assert 0 <= seed < 2 ** 32
    return seed


def make_env(which: str, rank: int):
    env = gym.make(id=which)
    env.seed(42 + rank)

    return env


def actual_datetime() -> str:
    """Returns the current data timestamp, formatted as follows: YearMonthDay-HourMinuteSecond"""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def np_normalize(x, epsilon=NP_EPS):
    return (x - np.mean(x)) / (np.std(x) + epsilon)


def discount_cumsum(x, discount: float):
    """Performs a cumulative discounted sum.
        - Source: https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/ppo/core.py#L45
    """
    return scipy.signal.lfilter([1.0], [1.0, float(-discount)], x[::-1], axis=0)[::-1]


def gae(rewards, values, gamma: float, lambda_: float):
    """Generalized Advantage Estimation (GAE) formula"""
    if lambda_ == 0.0:
        # special case: GAE(\gamma, 0)
        advantages = rewards[:-1] + gamma * values[1:] - values[:-1]

    elif lambda_ == 1.0:
        # special case: GAE(\gamma, 1)
        advantages = rewards_to_go(rewards, discount=gamma) - values[:-1]
    else:
        deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
        advantages = discount_cumsum(deltas, discount=gamma * lambda_)

    return advantages


def rewards_to_go(rewards, discount: float, decompose=False):
    if len(rewards) == 1:
        returns = rewards
    else:
        returns = discount_cumsum(rewards, discount=discount)[:-1]

    if decompose:
        returns_base, returns_exp = tf.map_fn(fn=decompose_number, elems=to_float(returns),
                                              dtype=(tf.float32, tf.float32))

        return tf.stack([returns_base, returns_exp], axis=1), returns

    return returns


def is_image(x) -> bool:
    """Checks whether some input [x] has a shape of the form (H, W, C)"""
    return len(x.shape) == 3


def is_vector(x) -> bool:
    """Checks whether some input [x] has a shape of the form (N, D) or (D,)"""
    return 1 <= len(x.shape) <= 2


def is_empty(x: Union[np.ndarray, tf.Tensor]) -> bool:
    """Checks whether an np.array or tf.Tensor `x` is empty or not"""
    return (x.shape[0] == 0) and (x.shape[-1] == 0)


def is_scalar(x: Union[int, float, tuple, list, np.ndarray, tf.Tensor]) -> bool:
    """Checks whether `x` is a scalar value, thus shapeless"""
    if isinstance(x, (int, float)) or np.isscalar(x) or (np.ndim(x) == 0):
        return True

    if isinstance(x, (tuple, list)):
        return len(np.array(x).shape) == 0

    if isinstance(x, np.ndarray):
        return len(x.shape) == 0

    if tf.is_tensor(x):
        return len(tf.shape(x)) == 0

    raise ValueError(f'Argument `x` must be [int, float, np.ndarray, tf.Tensor] not {type(x)}.')


def is_tensor_like(x: Union[np.ndarray, tf.Tensor]) -> bool:
    return tf.is_tensor(x) or (isinstance(x, np.ndarray) and len(x.shape) > 0)


def to_tuple(x: Union[int, float, bool, list, tuple]) -> tuple:
    if isinstance(x, (int, float, bool)):
        return tuple([x])

    if isinstance(x, list):
        return tuple(x)

    if not isinstance(x, tuple):
        raise ValueError(f'Argument must of type int, float, bool, list or tuple not {type(x)}.')

    return x


def to_dict_for_log(x: Union[dict, int, float, np.ndarray, tf.Tensor], name: str) -> dict:
    """Returns a dict from given `x` for logging purpose: e.g. the dimensions of x are distinguished etc"""
    # scalar (single) item
    if is_scalar(x):
        return {name: x}

    # multiple scalar items
    if isinstance(x, np.ndarray) or tf.is_tensor(x):
        if len(x.shape) == 1:
            return {f'{name}_{i}': a for i, a in enumerate(x)}
        else:
            return {f'{name}_{i}': np.mean(x[:, i]) for i in range(x.shape[-1])}

    # composite items
    assert isinstance(x, dict)
    d = {}

    for k, v in x.items():
        if is_scalar(v):
            d[f'{name}-{k}'] = v

        if isinstance(v, np.ndarray) or tf.is_tensor(v):
            if len(v.shape) == 1:
                d.update({f'{name}_{i}': a for i, a in enumerate(v)})
            else:
                d.update({f'{name}_{i}': np.mean(v[:, i]) for i in range(v.shape[-1])})

    return d


def depth_concat(*arrays):
    return np.concatenate(*arrays, axis=-1)


def clip(value, min_value, max_value):
    return min(max_value, max(value, min_value))


def polyak_averaging(model, target, alpha: float):
    for var, var_target in zip(model.trainable_variables, target.trainable_variables):
        value = alpha * var_target + (1.0 - alpha) * var

        var_target.assign(value, read_value=False)


def clip_gradients(gradients: list, norm: float) -> list:
    """Clips each gradient in the given list by w.r.t. their own norm"""
    return [tf.clip_by_norm(grad, clip_norm=norm) for grad in gradients]


@tf.function
def clip_gradients_global(gradients: List[tf.Tensor], norm: float) -> tuple:
    """Clip given list of gradients w.r.t their `global norm`"""
    return tf.clip_by_global_norm(gradients, clip_norm=norm)  # returns: (grads, g_norm)


def accumulate_gradients(grads1: list, grads2: Optional[list] = None) -> list:
    if grads2 is None:
        return grads1

    return [g1 + g2 for g1, g2 in zip(grads1, grads2)]


def average_gradients(gradients: list, n: int) -> list:
    assert n > 0
    if n == 1:
        return gradients

    n = float(n)
    return [g / n for g in gradients]


def decompose_number(num: float) -> Tuple[float, float]:
    """Decomposes a given number `n` in a `base` (b) and `exponent` (e), such that:
       - n = b * 10^e
       - NOTE: `b` in [-1.0, 1.0], and `e` in [0.0, +inf]

       Examples:
       - 2.34 = (0.234, 1), such that 0.234 * 10^1 = 2.34
       - 0.034 = (0.034, 0.0), such that 0.034 * 10^0 = 0.034
       - -12 = (-0.12, 2.0), such that -0.12 * 10^2 = -12
    """
    exponent = 0

    while abs(num) > 1.0:
        num /= 10.0
        exponent += 1

    return num, float(exponent)


def remove_keys(d: dict, keys: List[str]):
    for key in keys:
        d.pop(key, None)


class Timed:
    """Measures time. Usage:
        with Timed():
            code()
    """
    def __init__(self, msg='Code', silent=False):
        self.t0 = None
        self.msg = str(msg)
        self.is_silent = bool(silent)

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, *args):
        if not self.is_silent:
            print(f'{self.msg} took {round(time.time() - self.t0, 3)}s.')


class Registry:
    """A "Manager" class for easily retrieving registered classes:
        - Classes must be registered upon definition.
    """

    def __init__(self):
        self.registry = dict()
        self._cls_reg = dict()

    def register(self, name: str, class_):
        assert callable(class_)

        if name in self.registry:
            other_class = self.registry[name]

            if class_ == other_class:
                raise ValueError(f'Name "{name}" already registered for class: "{class_}".')
            else:
                raise ValueError(f'Trying to register the same name "{name}" for two different classes: 1. '
                                 f'{other_class}", 2. "{class_}".')

        self.registry[name] = class_
        self._cls_reg[class_] = name

    def is_registered(self, class_) -> bool:
        return class_ in self._cls_reg

    def retrieve(self, name: str) -> Callable:
        assert isinstance(name, str)

        if name not in self.registry:
            raise ValueError(f'Name "{name}" not registered.')

        return self.registry[name]

    def registered_name(self, class_) -> str:
        return self._cls_reg[class_]

    def registered_classes(self) -> list:
        return list(self.registry.items())


# -------------------------------------------------------------------------------------------------
# -- Plot utils
# -------------------------------------------------------------------------------------------------

def plot_images(images: list, title=None):
    """Plots a list of images, arranging them in a rectangular fashion"""
    num_plots = len(images)
    rows = round(math.sqrt(num_plots))
    cols = math.ceil(math.sqrt(num_plots))

    if title is not None:
        plt.title(str(title))

    for k, img in enumerate(images):
        plt.subplot(rows, cols, k + 1)
        plt.axis('off')
        plt.imshow(img)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def plot_parameter(parameter: DynamicType, iterations: int, initial_step=0, show=True):
    assert iterations > 0
    parameter = DynamicParameter.create(value=parameter)

    for _ in range(initial_step):
        parameter.on_episode()

    data = []
    for _ in range(iterations):
        data.append(parameter())
        parameter.on_episode()

    plt.plot(data)

    if show:
        plt.show()


def plot(colormap='Set3', **kwargs):  # Pastel1, Set3, tab20b, tab20c, ...
    """Colormaps: https://matplotlib.org/tutorials/colors/colormaps.html"""
    num_plots = len(kwargs.keys())
    cmap = plt.get_cmap(name=colormap)
    rows = round(math.sqrt(num_plots))
    cols = math.ceil(math.sqrt(num_plots))

    for k, (key, value) in enumerate(kwargs.items()):
        plt.subplot(rows, cols, k + 1)
        plt.plot(value, color=cmap(k + 1))
        plt.title(key)

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
        # spec[name] = (space.n,)
        spec[name] = (1,)

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


def space_to_flat_spec2(space: gym.Space, name: str) -> Dict[str, dict]:
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
        spec[name] = dict(shape=(space.n,), dtype=tf.int32)

    elif isinstance(space, spaces.MultiDiscrete):
        spec[name] = dict(shape=space.nvec.shape, dtype=tf.int32)

    elif isinstance(space, spaces.Box):
        spec[name] = dict(shape=space.shape, dtype=tf.float32,
                          low=space.low, high=space.high,
                          bounded_below=space.bounded_below, bounded_above=space.bounded_above)

    elif isinstance(space, spaces.Dict):
        for key, value in space.spaces.items():
            space_name = f'{name}_{key}'
            result = space_to_flat_spec(space=value, name=space_name)

            for k, v in result.items():
                if isinstance(v, dict):
                    spec[k] = v
                else:
                    spec[space_name] = result
                    break
    else:
        raise TypeError(f'`space` must be one of Box, Discrete, MultiDiscrete, or Dict, not {type(space)}!')

    return spec


# -------------------------------------------------------------------------------------------------
# -- TF utils
# -------------------------------------------------------------------------------------------------

def tf_enable_debug():
    tf.config.run_functions_eagerly(True)


@tf.function
def to_tensor(x, expand_axis=0):
    if isinstance(x, dict):
        return tf.nest.map_structure(lambda v: to_tensor(v, expand_axis), x)
    else:
        x = to_float(x)

    if isinstance(expand_axis, int):
        x = tf.expand_dims(x, axis=expand_axis)

    return x


def index_tensor(tensor, indices):
    """Index some `tensor` by some other tensor (`indices`)"""
    shape = (tensor.shape[0], 1)

    indices = tf.concat([
        tf.reshape(tf.range(start=0, limit=shape[0], dtype=tf.int32), shape),
        tf.reshape(tf.cast(indices, dtype=tf.int32), shape),
    ], axis=1)

    return tf.gather_nd(tensor, indices)


def tf_replace_nan(tensor, value=0.0, dtype=tf.float32):
    replacement = tf.constant(value, dtype=dtype, shape=tensor.shape)
    return tf.where(tensor == tensor, x=tensor, y=replacement)


def num_dims(tensor) -> tf.int32:
    """Returns the dimensionality (number of dimensions/axis) of the given tensor"""
    return tf.rank(tf.shape(tensor))


def mask_dict_tensor(tensor: dict, mask) -> dict:
    return {k: v[mask] for k, v in tensor.items()}


def concat_tensors(*tensors, axis=0) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
    assert len(tensors) > 0

    if isinstance(tensors[0], dict):
        return concat_dict_tensor(*tensors, axis=axis)

    return tf.concat(tensors, axis=axis)


def concat_dict_tensor(*dicts, axis=0) -> dict:
    assert len(dicts) > 0
    assert isinstance(dicts[0], dict)

    result = dicts[0]

    for i in range(1, len(dicts)):
        d = dicts[i]
        result = {k: tf.concat([v, d[k]], axis=axis) for k, v in result.items()}

    return result


def tf_chance(shape=(1,), lower=0.0, upper=1.0, seed=None):
    """Use to get random numbers between `lower` and `upper`"""
    return tf.random.uniform(shape=shape, minval=lower, maxval=upper, seed=seed)


def tf_count(tensor: tf.Tensor, value: Union[int, float, tf.Tensor]):
    """Counts occurrences of `value` within `tensor`."""
    return tf.reduce_sum(tf.cast(tensor == value, dtype=tf.int32))


def tf_normalize(x, eps=TF_EPS):
    """Normalizes some tensor `x` to 0-mean 1-stddev (aka standardization`)"""
    x = to_float(x)
    return (x - tf.math.reduce_mean(x)) / (tf.math.reduce_std(x) + eps)


def tf_sp_norm(x, eps=1e-3):
    """Sign-preserving normalization:
        - normalizes positive values of `x` independently from the negative one.
    """
    x = to_float(x)

    positives = x * to_float(x > 0.0)
    negatives = x * to_float(x < 0.0)
    return (positives / (tf.reduce_max(x) + eps)) + (negatives / -(tf.reduce_min(x) - eps))


def tf_magnitude_norm(x):
    """Magnitude normalization"""
    x = to_float(x)

    # decompose `x` such that `x = base * 10^exp`
    base, exp = tf.map_fn(fn=decompose_number, elems=x, dtype=(tf.float32, tf.float32))
    exp = tf.reshape(exp, shape=base.shape)

    return tf.where(base > 0.0, x=base + exp, y=base - exp)


def tf_minmax_norm(x, lower=0.0, upper=1.0, eps=TF_EPS):
    """Min-Max normalization, which scales `x` to be in range [lower, upper]"""
    x = to_float(x)
    x_min = tf.minimum(x) + eps
    return (x - x_min) / (tf.maximum(x) - x_min) * (upper - lower) + lower


def tf_shuffle_tensors(*tensors, indices=None):
    """Shuffles all the given tensors in the SAME way.
       Source: https://stackoverflow.com/questions/56575877/shuffling-two-tensors-in-the-same-order
    """
    assert len(*tensors) > 0

    if indices is None:
        indices = tf.range(start=0, limit=tf.shape(tensors[0])[0], dtype=tf.int32)
        indices = tf.random.shuffle(indices)

    return [tf.gather(t, indices) for t in tensors]


def data_to_batches(tensors: Union[list, dict, tuple], batch_size: int, shuffle_batches=False, take: int = None,
                    drop_remainder=False, map_fn=None, prefetch_size=2, num_shards=1, skip=0, seed=None, shuffle=False,
                    filter_fn: Callable[[Any], bool] = None):
    """Transform some tensors data into a dataset of mini-batches"""
    dataset = tf.data.Dataset.from_tensor_slices(tensors).skip(count=skip)

    if callable(filter_fn):
        dataset = dataset.filter(predicate=filter_fn)

    if isinstance(take, int) and take > 0:
        dataset = dataset.take(count=take)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size, seed=seed, reshuffle_each_iteration=True)

    if num_shards > 1:
        # "observation skip trick" with tf.data.Dataset.shard()
        ds = dataset.shard(num_shards, index=0)

        for shard_index in range(1, num_shards):
            shard = dataset.shard(num_shards, index=shard_index)
            ds = ds.concatenate(shard)

        dataset = ds

    if map_fn is not None:
        # 'map_fn' is mainly used for 'data augmentation' and 'pre-processing'
        dataset = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE,
                              deterministic=True)

    # TODO: add `num_parallel_calls=AUTOTUNE` when updating tf
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    if shuffle_batches:
        dataset = dataset.shuffle(buffer_size=batch_size, seed=seed, reshuffle_each_iteration=True)

    return dataset.prefetch(buffer_size=prefetch_size)


def tf_to_scalar_shape(tensor):
    return tf.reshape(tensor, shape=[])


def assert_shapes(a, b):
    assert tf.shape(a) == tf.shape(b)


def softplus(value=1.0):
    @tf.function
    def activation(x):
        return tf.nn.softplus(x) + value

    return activation


# @tf.function
def swish6(x):
    return tf.minimum(tf.nn.swish(x), 6.0)


def dsilu(x):
    """dSiLu activation function (i.e. the derivative of SiLU/Swish).
       Paper: Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning
    """
    sigma_x = tf.nn.sigmoid(x)
    return sigma_x * (1.0 + x * (1.0 - sigma_x))


@tf.function
def batch_norm_relu6(layer: tf.keras.layers.Layer):
    """BatchNormalization + ReLU6, use as activation function"""
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.nn.relu6(layer)
    return layer


@tf.function
def lisht(x):
    """Non-Parametric Linearly Scaled Hyperbolic Tangent Activation Function
       Sources:
        - https://www.tensorflow.org/addons/api_docs/python/tfa/activations/lisht
        - https://arxiv.org/abs/1901.05894
    """
    return tf.multiply(x, tf.nn.tanh(x))


@tf.function
def mish(x):
    """A Self Regularized Non-Monotonic Neural Activation Function
       Source:
        - https://www.tensorflow.org/addons/api_docs/python/tfa/activations/mish
    """
    return tf.multiply(x, tf.nn.tanh(tf.nn.softplus(x)))


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


def tf_dot_product(x, y, axis=0, keepdims=False):
    return tf.reduce_sum(tf.multiply(x, y), axis=axis, keepdims=keepdims)


@tf.function
def stop_gradient(args: tuple, **kwargs) -> tuple:
    return tf.nest.map_structure(tf.stop_gradient, args, **kwargs)


def tf_flatten(x):
    """Reshapes the given input as a 1-D array"""
    return tf.reshape(x, shape=[-1])


def tf_norm(items: list, **kwargs) -> list:
    return [tf.norm(x, **kwargs) for x in items]


def tf_global_norm(values: list, from_norms=True, **kwargs):
    if not from_norms:
        values = tf_norm(values, **kwargs)

    return tf.sqrt(tf.reduce_sum([norm * norm for norm in values]))


@tf.function
def huber_loss(errors: tf.Tensor, kappa=1.0) -> tf.Tensor:
    """The Huber loss is defined via two cases:
          - case_one: |errors| <= kappa (cutoff)
          - case_two: |errors| > kappa"""
    abs_errors = tf.abs(errors)

    case_one = to_float(abs_errors <= kappa) * 0.5 * tf.square(errors)  # MSE
    case_two = to_float(abs_errors > kappa) * kappa * (abs_errors - 0.5 * kappa)  # k * (|x| - 1/2 k)

    return case_one + case_two


def tf_normal_cdf(x) -> tf.Tensor:
    """Standard-Normal cumulative distribution function (CDF)
        - Source: https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function
    """
    return 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))


def tf_normal_inverse_cdf(x) -> tf.Tensor:
    """Standard-Normal inverse cumulative distribution function (CDF)
        - Source: https://en.wikipedia.org/wiki/Normal_distribution#Quantile_function
    """
    return tf.sqrt(2.0) * tf.math.erfinv(2.0 * x - 1.0)


# class DynamicArray:
#     """Dynamic-growing np.array"""
#
#     def __init__(self, shape: tuple, max_capacity: int, min_capacity=16, dtype=np.float32):
#         assert min_capacity > 0
#         assert max_capacity >= min_capacity
#
#         self.elem_shape = shape
#         self.dtype = dtype
#
#         self.min_capacity = min_capacity
#         self.max_capacity = max_capacity
#         self.current_capacity = self.min_capacity
#
#         self.array = self._allocate(capacity=self.min_capacity)
#         self.index = 0
#
#     @property
#     def shape(self):
#         return self.array.shape
#
#     def grow(self):
#         self.current_capacity = min(2 * self.current_capacity, self.max_capacity)
#
#         old_array = self.array
#         self.array = self._allocate(capacity=self.current_capacity)
#         self._copy(array=old_array)
#
#         del old_array
#
#     # def insert(self, data, index: int):
#     #     assert 0 <= index < self.current_capacity
#     #
#     #     self.array[index] = self._to_numpy(data)
#
#     def append(self, data):
#         if self.index >= self.current_capacity:
#             assert self.index < self.max_capacity, 'DynamicArray is full!'
#             print('-- grow --')
#             self.grow()
#
#         self.array[self.index] = self._to_numpy(data)
#         self.index += 1
#
#     def to_tensor(self, dtype=tf.float32) -> tf.Tensor:
#         return tf.constant(self.array[:self.index], dtype=dtype)
#
#     def clean(self):
#         pass
#
#     def __repr__(self):
#         return self.array.__repr__()
#
#     def _to_numpy(self, data):
#         data = np.asarray(data, dtype=self.dtype)
#         return np.reshape(data, newshape=self.elem_shape)
#
#     def _copy(self, array):
#         self.array[:array.shape[0]] = array
#
#     def _allocate(self, capacity: int):
#         return np.zeros(shape=(capacity,) + self.elem_shape, dtype=self.dtype)


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


def load_traces(traces_dir: str, max_amount: Optional[int] = None, shuffle=False, offset=0):
    assert offset >= 0

    if shuffle:
        trace_names = file_names(traces_dir, sort=False)
        random.shuffle(trace_names)
    else:
        trace_names = file_names(traces_dir, sort=True)

    if max_amount is None:
        max_amount = np.inf

    for i in range(offset, len(trace_names)):
        name = trace_names[i]
        if i >= max_amount:
            return
        print(f'loading {name}...')
        yield np.load(file=os.path.join(traces_dir, name))


def count_traces(traces_dir: str) -> int:
    """Returns the number of traces available at the given folder."""
    return len(file_names(traces_dir, sort=False))


def unpack_trace(trace: dict, unpack=True) -> Union[tuple, dict]:
    """Reads a trace (i.e. a dict-like object created by np.load()) and unpacks it as a tuple
       (state, action, reward, done).
       - When `unpack is False` the (processed) trace dict is returned.
    """
    trace_keys = trace.keys()
    trace = {k: trace[k] for k in trace_keys}  # copy

    for name in ['state', 'action']:
        # check if state/action space is simple (array, i.e sum == 1) or complex (dict of arrays)
        if sum(k.startswith(name) for k in trace_keys) == 1:
            continue

        # select keys of the form 'state_xyz', then build a dict(state_xyz=trace['state_xyz'])
        keys = filter(lambda k: k.startswith(name + '_'), trace_keys)
        trace[name] = {k: trace[k] for k in keys}

    if 'done' not in trace:
        trace['done'] = None

    if unpack:
        return trace['state'], trace['action'], to_float(trace['reward']), trace['done']

    # remove fields of the form `state_x`, `action_y`, ...
    for key in trace_keys:
        if 'state' in key or 'action' in key:
            if key != 'state' and key != 'action':
                trace.pop(key)

    return trace


def copy_folder(src: str, dst: str):
    """Source: https://stackoverflow.com/a/31039095"""
    dir_util.copy_tree(src, dst)


def remove_folder(path: str, **kwargs):
    dir_util.remove_tree(path, **kwargs)


# -------------------------------------------------------------------------------------------------
# -- Statistics
# -------------------------------------------------------------------------------------------------

def tf_explained_variance(x, y, eps=TF_EPS) -> float:
    """Computes fraction of variance that `x` explains about `y`.
       Interpretation:
            - ev = 0  =>  might as well have predicted zero
            - ev = 1  =>  perfect prediction
            - ev < 0  =>  worse than just predicting zero

        - Source: https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/math_util.py#L25
    """
    explained_var = 1.0 - (tf.math.reduce_variance(y - x) / (tf.math.reduce_variance(y) + eps))
    return tf.stop_gradient(explained_var)


def tf_residual_variance(x: tf.Tensor, y: tf.Tensor, eps=TF_EPS):
    residual_var = tf.math.reduce_variance(y - x) / (tf.math.reduce_variance(y) + eps)
    return tf.stop_gradient(residual_var)


# TODO: possible bug about logging 1-timestep long envs
class SummaryProcess(mp.Process):
    """Easy and efficient tf.summary with multiprocessing"""

    def __init__(self, queue: mp.Queue, stop_event: mp.Event, name=None, folder='logs', keys: List[str] = None,
                 log_every=10, flush_millis=30, max_queue=20):
        assert isinstance(name, str)
        super().__init__(daemon=True)

        self.steps = dict()
        self.queue = queue
        self.stop_event = stop_event
        self.log_freq = 1 if log_every <= 1 else int(log_every)

        # filters what to log
        if isinstance(keys, list):
            self.allowed_keys = {k: True for k in keys}
        else:
            self.allowed_keys = None

        self.summary_dir = os.path.join(folder, name, actual_datetime())
        self.tf_summary_writer = None
        self.summary_args = dict(flush_millis=flush_millis, max_queue=max_queue)

    def run(self):
        import tensorflow as tf  # import here, and lazy tf.summary.create is necessary for multiprocessing to work

        if self.tf_summary_writer is None:
            self.tf_summary_writer = tf.summary.create_file_writer(self.summary_dir, **self.summary_args)

        # wait for stuff to be logged
        while not self.stop_event.is_set():

            with self.tf_summary_writer.as_default():

                while not self.queue.empty():
                    try:
                        self.log(**self.queue.get())
                    except Exception as err:
                        print(err)

                if self.stop_event.is_set():
                    self.tf_summary_writer.close()

        self.tf_summary_writer.close()  # maybe redundant

    def log(self, average=False, **kwargs):
        for key, value in kwargs.items():
            if not self.should_log_key(key):
                continue

            # if key not in self.steps:
            #     self.steps[key] = 0
            #
            # step = self.steps[key]
            step = self.steps.setdefault(key, 0)
            value = tf.convert_to_tensor(value)

            if ('_hist' in key) or ('weight-' in key) or ('bias-' in key):
                if self.should_log(step):
                    tf.summary.histogram(name=key, data=value, step=step)

                self.steps[key] += 1

            elif 'image_' in key:
                if self.should_log(step):
                    tf.summary.image(name=key, data=tf.concat(value, axis=0), step=step)

                self.steps[key] += 1
            else:
                self._scalar_summary(average, key, value, step)

    def _scalar_summary(self, average: bool, key: str, value, step: int):
        value = tf.squeeze(value)

        if ('action' in key) or ('prob' in key):
            if len(value.shape) == 0:
                # scalar (=> one action)
                if self.should_log(step):
                    tf.summary.scalar(name=key, data=value, step=step)

                self.steps[key] += 1

            elif len(value.shape) == 1:
                # (N,)-shaped array (=> 1-d actions)
                if average:
                    tf.summary.scalar(name=key, data=tf.reduce_mean(value), step=step)
                    self.steps[key] += 1
                else:
                    for i, v in enumerate(value):
                        if self.should_log(step + i):
                            tf.summary.scalar(name=key, data=v, step=step + i)

                    self.steps[key] += value.shape[0]

            elif len(value.shape) == 2:
                # (M, N)-shaped array of actions (=> N-dim actions)
                if average:
                    value = tf.reduce_mean(value, axis=0, keepdims=True)  # mean across batch-axis

                # loop over action-axis (i)
                for i in range(value.shape[1]):
                    action_key = f'{key}-{i}'

                    if action_key not in self.steps:
                        self.steps[action_key] = 0

                    step = self.steps[action_key]

                    # loop over batch-axis (j)
                    for j in range(value.shape[0]):
                        if average or self.should_log(step + j):
                            tf.summary.scalar(name=action_key, data=value[j][i], step=step + j)

                    self.steps[action_key] += value.shape[0]

        elif average or len(value.shape) == 0:
            value = tf.reduce_mean(value)

            if ('reward' in key) and (key != 'episode_reward'):
                if self.should_log(step):
                    tf.summary.scalar(name=key, data=value, step=step)
            else:
                tf.summary.scalar(name=key, data=value, step=step)

            self.steps[key] += 1
        else:
            self.steps[key] += value.shape[0]

            for i, v in enumerate(value):
                if self.should_log(step + i):
                    tf.summary.scalar(name=key, data=tf.reduce_mean(v), step=step + i)

    def should_log(self, step: int) -> bool:
        return step % self.log_freq == 0

    def should_log_key(self, key: str) -> bool:
        if self.allowed_keys is None:
            return True

        return key in self.allowed_keys

    def close(self):
        # TODO: should remaining logs be written?
        # consume remaining logs
        while not self.queue.empty():
            self.queue.get()

        if self.is_alive():
            self.join()
            self.terminate()


# TODO: deprecate
class IncrementalStatistics:
    """Compute mean, variance, and standard deviation incrementally."""
    def __init__(self, epsilon=TF_EPS, max_count=10e8):
        self.mean = 0.0
        self.variance = 0.0
        self.std = 0.0
        self.count = 0

        self.eps = epsilon
        self.max_count = int(max_count)  # fix: cannot convert 10e8 to EagerTensor of type int32

    def update(self, x, normalize=False):
        old_mean = self.mean
        new_mean = tf.reduce_mean(x)
        m = self.count
        n = tf.shape(x)[0]
        c1 = m / (m + n)
        c2 = n / (m + n)

        # more numerically stable than `c3 = (m * n) / (m + n + eps) ** 2` (no square at the denominator,
        # does not go to infinite but could became zero when m -> inf, so `m` should be clipped as well)
        c3 = 1.0 / ((m / n) + 2.0 + (n / m))

        self.mean = c1 * old_mean + c2 * new_mean
        self.variance = c1 * self.variance + c2 * tf.math.reduce_variance(x) + c3 * (old_mean - new_mean) ** 2 + self.eps
        self.std = tf.sqrt(self.variance)

        # limit accumulating values to avoid numerical instability
        self.count = min(self.count + n, self.max_count)

        if normalize:
            return self.normalize(x)

    def normalize(self, values, eps=TF_EPS):
        return to_float((values - self.mean) / (self.std + eps))

    def set(self, mean: float, variance: float, std: float, count: int):
        self.mean = mean
        self.variance = variance
        self.std = std
        self.count = count

    def as_dict(self) -> dict:
        return dict(mean=np.float(self.mean), variance=np.float(self.variance),
                    std=np.float(self.std), count=np.int(self.count))
