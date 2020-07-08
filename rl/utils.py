import os
import gym
import math
import numpy as np
import scipy.signal
import tensorflow as tf
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
import imgaug as ia

from typing import Union, List, Dict, Tuple
from datetime import datetime

from gym import spaces


def to_tensor(x, expand_axis=0):
    if isinstance(x, dict):
        for k, v in x.items():
            x[k] = tf.expand_dims(tf.convert_to_tensor(v), axis=expand_axis)
    else:
        x = tf.convert_to_tensor(x)
        x = tf.expand_dims(x, axis=expand_axis)
    return x


def tf_normalize(x):
    """Normalizes some tensor x to 0-mean 1-stddev"""
    return (x - tf.math.reduce_mean(x)) / tf.math.reduce_std(x)


def np_normalize(x, epsilon=np.finfo(np.float32).eps):
    return (x - np.mean(x)) / (np.std(x) + epsilon)


def discount_cumsum(x, discount: float):
    """Source: https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/ppo/core.py#L45"""
    return scipy.signal.lfilter([1.0], [1.0, float(-discount)], x[::-1], axis=0)[::-1]


def gae(rewards, values, gamma: float, lambda_: float, normalize=False):
    deltas = rewards[:-1] + tf.math.multiply(values[1:], gamma) - values[:-1]
    advantages = discount_cumsum(deltas, discount=gamma * lambda_)

    if normalize:
        advantages = tf_normalize(advantages)

    return tf.cast(advantages, dtype=tf.float32)


def rewards_to_go(rewards, discount: float, normalize=False):
    returns = discount_cumsum(rewards, discount=discount)[:-1]

    if normalize:
        returns = np_normalize(returns)

    return returns


def data_to_batches(tensors: Union[List, Tuple], batch_size: int, shuffle=False, seed=None,
                    drop_remainder=False, map_fn=None):
    """Transform some tensors data into a dataset of mini-batches"""
    dataset = tf.data.Dataset.from_tensor_slices(tensors)

    if map_fn is not None:
        dataset = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE,
                              deterministic=True)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    if shuffle:
        return dataset.shuffle(buffer_size=batch_size, seed=seed)

    return dataset


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


def tf_to_scalar_shape(tensor):
    return tf.reshape(tensor, shape=[])


def assert_shapes(a, b):
    assert tf.shape(a) == tf.shape(b)


def space_to_flat_spec(space: gym.Space, name: str) -> Dict[str, tuple]:
    """From a gym.Space object returns a flat dictionary str -> tuple.
       Naming convention:
         - If space is Box or Discrete, it returns 'dict(name=shape)'
         - If space is Dict (not nested), it returns 'dict(name_x=shape_x, name_y=shape_y)'
            considering 'x' and 'y' be component of space.
         - With further nesting, dict keys' names got created using the above two rules.
           In this way each key (name) uniquely identifies a (sub-)component of the space.
    """
    spec = dict()

    if isinstance(space, spaces.Discrete):
        spec[name] = (space.n,)

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
        raise ValueError('space must be one of Box, Discrete, or Dict')

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

    assert isinstance(space, spaces.Dict)

    spec = dict()
    for name, space in space.spaces.items():
        # use recursion to handle arbitrary nested Dicts
        spec[name] = space_to_spec(space)

    return spec


def is_image(x) -> bool:
    """Checks whether some input [x] has a shape of the form (H, W, C)"""
    return len(x.shape) == 3


def is_vector(x) -> bool:
    """Checks whether some input [x] has a shape of the form (N, D) or (D,)"""
    return 1 <= len(x.shape) <= 2


def depth_concat(*arrays):
    return np.concatenate(*arrays, axis=-1)


def tf_01_scaling(x):
    x -= tf.reduce_min(x)
    x /= tf.reduce_max(x)
    return x


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
# -- File Utils
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


# -------------------------------------------------------------------------------------------------
# -- Statistics
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

            if hasattr(value, '__iter__'):
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


# -------------------------------------------------------------------------------------------------
# -- Data Augmentation
# -------------------------------------------------------------------------------------------------

class AugmentationPipeline:
    def __init__(self, strength=1.0, random_order=True, seed=None):
        assert 0.0 < strength <= 1.0
        self.alpha = strength

        if seed is not None:
            ia.seed(seed)

        self.augmentations = iaa.Sequential([
            iaa.ChannelShuffle(p=0.35 * strength),

            # Contrast
            iaa.LinearContrast(alpha=self._weaken(0.4, 1.6)),

            # Brightness
            iaa.MultiplyBrightness(mul=self._weaken(0.7, 1.3)),

            # Tone
            iaa.MultiplyHueAndSaturation(self._weaken(0.5, 1.5)),

            # Blur
            iaa.SomeOf((0, 2), [
                iaa.GaussianBlur(sigma=(0.0, 2.0 * strength)),
                iaa.MotionBlur(k=(3, round(7 * strength)), angle=self._balance(-45, 45)),
            ]),

            # Noise
            iaa.SomeOf((0, 2), [
                iaa.AdditiveGaussianNoise(loc=0.0, scale=self._balance(0.01, 0.2 * 255)),
                iaa.SaltAndPepper(p=self._balance(0.01, 0.15)),
            ]),

            # Masking
            iaa.SomeOf((0, 2), [
                iaa.Cutout(nb_iterations=(0, 2), cval=0),
                iaa.CoarseDropout(p=self._balance(0.001, 0.05), size_percent=self._balance(0.02, 0.25)),
            ])
        ], random_order=random_order)

    def augment(self, images):
        return self.augmentations(images=images)

    def _weaken(self, a: float, b: float, value=1.0) -> tuple:
        a += abs(value - a) * (1.0 - self.alpha)
        b -= abs(value - b) * (1.0 - self.alpha)
        return a, b

    def _balance(self, a: float, b: float, cast=None) -> tuple:
        a = a * self.alpha
        b = b * self.alpha

        if cast == 'int':
            return round(a), round(b)

        return a, b
