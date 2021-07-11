"""Environments based on building decision trees (DTs)"""

import gym
import numpy as np
import tensorflow as tf

from typing import Tuple, Dict, Union, Callable
from sklearn.model_selection import train_test_split

from rl import utils
from rl.environments.tree.trees import TreeClassifier, TreeRegressor


class TreeEnv(gym.Env):
    def __init__(self, seed=utils.GLOBAL_SEED):
        self.random: np.random.Generator = None
        self._seed = None
        self.seed(seed)

    def seed(self, seed=utils.GLOBAL_SEED):
        self._seed = seed
        self.random = utils.get_random_generator(seed=self._seed)


# TODO: extend to `multi-label` and `multi-class`?
# TODO: make a generic TreeEnv (without specific tree type), then subclass
class TreeClassifierEnv(TreeEnv):
    """Gym Environment in which the agent has to build a decision tree to classify some dataset."""
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, batch_size: int, validation_size=0.2, aggregation='avg',
                 reward: Union[str, dict, Callable] = 'accuracy', num_batch_reward=np.inf, seed=utils.GLOBAL_SEED,
                 debug=True, pad: Union[None, int, float] = 0.0, **kwargs):
        assert batch_size >= 1
        assert len(x_data.shape) >= 2
        assert 1 <= len(y_data.shape) <= 2
        assert isinstance(aggregation, str)
        assert isinstance(num_batch_reward, (float, int)) and num_batch_reward >= 1

        super().__init__(seed)

        self.debug = bool(debug)
        self.kwargs = kwargs
        self.batch_size = int(batch_size)
        self.reward_fn = self._get_reward_fn(reward)
        self.aggregate = self._get_aggregation_fn(aggregation)

        self.x_train, self.x_val, \
        self.y_train, self.y_val = train_test_split(x_data, y_data, test_size=validation_size, random_state=self._seed)

        # prepare validation-set for computing rewards
        ds = tf.data.Dataset.from_tensor_slices(tensors=(self.x_val, self.y_val))
        ds = ds.shuffle(buffer_size=self.batch_size, seed=self._seed, reshuffle_each_iteration=True)
        ds = ds.repeat(count=-1).batch(self.batch_size)
        self.validation_set = ds.prefetch(2)  # TODO: prefetch dataset does help?

        # determine number of batches used to determine each reward
        self.num_validation_batches = np.ceil(self.x_val.shape[0] / self.batch_size)
        self.num_validation_batches = np.minimum(num_batch_reward, self.num_validation_batches)

        # TODO: should the min/max be computed only on the training-set?
        self.x_low = np.minimum(np.min(self.x_train, axis=0), np.min(self.x_val, axis=0))
        self.x_high = np.maximum(np.max(self.x_train, axis=0), np.max(self.x_val, axis=0))

        self.tree = TreeClassifier(x_train=self.x_train, y_train=self.y_train, **self.kwargs)  # root
        self.node = self.tree  # current node
        self.lifo = [self.tree]  # Last-In First-out queue to determine the growth order of the tree

        self.num_features = 3 + self.tree.max_split + len(self.tree.classes)  # 3: depth, feature_index, num children

        if isinstance(pad, (int, float)):
            self.should_pad = True
            self.pad_value = float(pad)
        else:
            self.should_pad = False

    @property
    def action_space(self) -> gym.spaces.Dict:
        # TODO: provide option to have "unbounded" split values?
        return gym.spaces.Dict(num_splits=gym.spaces.Discrete(n=self.tree.max_split),
                               feature_index=gym.spaces.Discrete(n=self.x_train.shape[-1]),
                               split_values=gym.spaces.Box(low=np.min(self.x_low), high=np.max(self.x_high),
                                                           shape=(self.tree.max_split,)))

    @property
    def observation_space(self):
        return gym.spaces.Dict(x=gym.spaces.Box(shape=(self.batch_size,) + self.x_train.shape[1:],
                                                low=-np.inf, high=np.inf),
                               y=gym.spaces.Box(shape=(self.batch_size,) + self.y_train.shape[1:],
                                                low=0.0, high=np.max(self.y_train)),
                               tree=gym.spaces.Box(shape=(self.tree.max_depth + 1, self.num_features),
                                                   low=-np.inf, high=np.inf))

    def step(self, action: Dict[str, np.ndarray]):
        num_splits = int(np.squeeze(action['num_splits']))  # TODO: minimum value should be one
        index = int(np.squeeze(action['feature_index']))

        self.node.split(index=index, values=action['split_values'][:num_splits])

        if self.node.depth + 1 <= self.tree.max_depth:
            for child in self.node.children:
                if not child.is_leaf:
                    self.lifo.insert(0, child)

        self.node = self.lifo.pop(0)

        return self.observation(), self.reward(), self.done(), self.info()

    def reset(self):
        if self.debug:
            self.tree.print()
            input('Press ENTER to continue...')

        self.tree = TreeClassifier(x_train=self.x_train, y_train=self.y_train, **self.kwargs)
        self.node = self.tree
        self.lifo = [self.tree]

        return self.observation()

    def observation(self) -> dict:
        x, y = self.sample()

        if self.should_pad:
            pad_size = self.batch_size - x.shape[0]

            if pad_size > 0:
                pad_x = np.full(shape=(pad_size, x.shape[-1]), fill_value=self.pad_value, dtype=np.float32)
                # TODO: generalize shape
                pad_y = np.full(shape=(pad_size,), fill_value=self.pad_value, dtype=np.float32)

                x = np.concatenate([x, pad_x], axis=0)
                y = np.concatenate([y, pad_y], axis=0)

        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)

        if len(y.shape) == 2:
            y = np.reshape(y, newshape=(1, y.shape[1], -1))

        return dict(x=x, y=y, tree=self._tree_obs())

    def reward(self) -> float:
        rewards = []

        for (x_batch, y_batch) in self.validation_set.take(count=self.num_validation_batches):
            y_pred = self.tree.predict(batch=x_batch, sort=False, debug=True)
            y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

            rewards.append(self.reward_fn(y_batch, y_pred))

        return self.aggregate(rewards)

    def done(self) -> bool:
        # the episode terminates when there are no more nodes to be growth
        return len(self.lifo) == 0

    def info(self) -> dict:
        # TODO: think about useful `info` to be provided?
        return {}

    def render(self, mode='human'):
        if self.debug:
            self.tree.print()
            input('Press ENTER to continue...')

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns a batch of data from the current node"""
        if self.node.num_samples <= self.batch_size:
            x = self.node.x_train
            y = self.node.y_train
        else:
            indices = self.random.choice(self.node.num_samples, size=self.batch_size, replace=False)
            x = self.node.x_train[indices]
            y = self.node.y_train[indices]

        return x, y

    @staticmethod
    def _get_reward_fn(identifier) -> Callable:
        if isinstance(identifier, (str, dict)):
            return tf.keras.metrics.get(identifier)

        elif callable(identifier):
            return identifier
        else:
            raise ValueError(f'Cannot define a reward function of type: {type(identifier)}')

    @staticmethod
    def _get_aggregation_fn(method):
        method = method.lower()

        if method in ['mean', 'avg', 'average']:
            return np.mean

        elif method == 'median':
            return np.median

        elif method in ['min', 'minimum']:
            return np.min

        elif method in ['max', 'maximum']:
            return np.max
        else:
            raise ValueError(f'Not supported method "{method}" for aggregating rewards.')

    def _tree_obs(self):
        if self.tree.is_empty():
            return np.zeros(shape=(1, self.tree.max_depth + 1, self.num_features), dtype=np.float32)

        tree_obs = self.node.as_features(split_size=self.tree.max_split)
        missing_nodes = self.tree.max_depth + 1 - tree_obs.shape[0]

        if missing_nodes > 0:
            # pad with leading zeros
            padding = np.zeros(shape=(missing_nodes, self.num_features), dtype=np.float32)
            return np.concatenate([tree_obs, padding], axis=0)

        return np.expand_dims(tree_obs, axis=0)


class TreeRegressorEnv(TreeEnv):
    pass
