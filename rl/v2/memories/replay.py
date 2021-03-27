
import numpy as np
import tensorflow as tf

from rl import utils
from rl.parameters import DynamicParameter
from rl.v2.memories import Memory

from typing import Union


class ReplayMemory(Memory):

    def get_batch(self, batch_size: int, **kwargs) -> dict:
        return self.sample(batch_size, **kwargs)

    # TODO: should sample with replacement?
    def sample(self, batch_size: int, seed=None) -> dict:
        """Samples a batch of transitions (without replacement)"""
        batch = dict()

        # random indices
        # indices = tf.range(start=0, limit=self.current_size, dtype=tf.int32)
        # indices = tf.random.shuffle(indices, seed=seed)[:batch_size]

        # indices = np.random.choice(self.current_size, size=batch_size, replace=True)
        indices = np.random.choice(self.current_size, size=batch_size, replace=False)

        for key, value in self.data.items():
            batch[key] = self._gather(value, indices)

        return batch

    def _gather(self, value, indices: tf.Tensor) -> Union[tf.Tensor, dict]:
        if not isinstance(value, dict):
            return tf.gather(value, indices)

        tensors = dict()

        for k, v in value.items():
            tensors[k] = self._gather(value=v, indices=indices)

        return tensors


class PrioritizedMemory(ReplayMemory):
    """Prioritized Experience Replay (PER) memory, proportional-variant
        - Based on https://github.com/BY571/Soft-Actor-Critic-and-Extensions/blob/master/SAC_PER.py
    """

    def __init__(self, *args, alpha: DynamicParameter, beta: DynamicParameter, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.beta = beta
        self.priorities = np.ones(shape=self.size, dtype=np.float32)

        self.indices = None
        self.weights = None
        self.td_error = None

    def store(self, transition: dict):
        super().store(transition)
        self.priorities[self.index - 1] = self.priorities[:self.index].max()

    # TODO: should sample with replacement?
    def sample(self, batch_size: int, seed=None) -> dict:
        """Samples a batch of transitions"""
        batch = dict()
        size = self.current_size

        # init td-error here (since `batch_size` may vary - should not!)
        # Also, td-error is a tf.Variable in order to deal with SymbolicTensor annoyances
        self.td_error = self._get_var(batch_size)

        # sample transitions according to priority
        prob = self.priorities[:size] ** float(self.alpha())
        prob /= prob.sum()
        prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)

        # TODO: try with "replacement"
        indices = np.random.choice(size, size=batch_size, replace=False, p=prob)
        # indices = np.random.choice(size, size=batch_size, replace=True, p=prob + utils.NP_EPS)
        self.indices = indices

        for key, value in self.data.items():
            batch[key] = self._gather(value, indices)

        # Importance-sampling weights
        weights = (size * prob[indices]) ** float(-self.beta())
        weights /= weights.max()

        # TODO: prevent user to use "weights" spec, when creating prioritized mem
        batch['weights'] = tf.expand_dims(weights, axis=-1)
        return batch

    @staticmethod
    def _get_var(size) -> tf.Variable:
        return tf.Variable(tf.zeros(size,), trainable=False, dtype=tf.float32)

    def update_priorities(self, eps=utils.NP_EPS):
        # adding an `eps` ensures probabilities to be non-zero
        self.priorities[self.indices] = np.abs(self.td_error.value()) + eps


class EmphasizingMemory(ReplayMemory):
    """Emphasizing Experience Replay (ERE) memory
        - Based on https://github.com/BY571/Soft-Actor-Critic-and-Extensions/blob/master/SAC_ERE_PER.py
    """

    def __init__(self, *args, eta_min=0.996, eta_max=1.0, episode_length=500, sampling_min=0.05, **kwargs):
        assert eta_min < eta_max
        assert 0 < eta_min <= 1
        assert 0 < eta_max <= 1
        assert episode_length > 0
        assert 0.0 < sampling_min <= 1.0

        super().__init__(*args, **kwargs)

        self.eta_min = float(eta_min)  # eta_0
        self.delta_eta = float(eta_max) - self.eta_min  # eta_T - eta_0

        self.timestep = 0  # t
        self.k = 0  # counts each time `sample()` is called before `reset()`
        self.episode_length = int(episode_length)

        self.sampling_min = int(self.size * sampling_min)  # c_min

    @property
    def eta(self) -> float:
        return self.eta_min + self.delta_eta * (self.timestep / self.episode_length)

    def sample(self, batch_size: int, seed=None) -> dict:
        batch = dict()
        size = self.current_size
        self.k += 1

        sampling_range = int(size * self.eta ** (self.k * (self.episode_length / self.timestep)))  # c_k
        sampling_range = max(sampling_range, max(batch_size, self.sampling_min))  # at least batch size

        if not self.is_full() or sampling_range <= self.index:
            # get indices associated to more recent experience (thus, start = size - c_k)
            indices = tf.range(start=size - sampling_range, limit=size, dtype=tf.int32)
        else:
            # `full and sampling_range > self.index` => two range of indices
            k = self.index - sampling_range  # k < 0

            range1 = tf.range(start=0, limit=self.index, dtype=tf.int32)
            range2 = tf.range(start=self.size - abs(k) - 1, limit=self.size, dtype=tf.int32)

            indices = tf.concat([range1 + range2], axis=0)
            assert sampling_range == indices.shape[0]

        # indices = tf.random.shuffle(indices, seed=seed)[:batch_size]
        indices = np.random.choice(indices, size=batch_size)

        for key, value in self.data.items():
            batch[key] = self._gather(value, indices)

        return batch

    def store(self, transition: dict):
        super().store(transition)
        self.timestep += 1

    def reset(self):
        self.k = 0
        self.timestep = 0

    def clear(self):
        super().clear()
        self.reset()
