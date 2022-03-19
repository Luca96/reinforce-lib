
import numpy as np
import tensorflow as tf

from rl import utils
from rl.parameters import DynamicParameter
from rl.v2.memories import Memory

from typing import Union


class ReplayMemory(Memory):

    def get_batch(self, batch_size: int, **kwargs) -> dict:
        batch = self.sample(batch_size)

        # retro-compatibility for ReplayMemory with DQN objective (see networks.q.DQN.objective(...))
        if 'return' not in batch:
            batch['return'] = batch['reward']

        return batch

    def sample(self, batch_size: int) -> dict:
        """Samples a batch of transitions (without replacement)"""
        batch = dict()

        # random indices
        indices = self.random.choice(self.current_size, size=batch_size, replace=False)

        for key, value in self.data.items():
            batch[key] = self._gather(value, indices)

        return batch

    def _gather(self, value: np.ndarray, indices: np.ndarray) -> Union[np.ndarray, dict]:
        if not isinstance(value, dict):
            # return tf.gather(value, indices)
            return value[indices]

        return {k: self._gather(value=v, indices=indices) for k, v in value.items()}


class NStepMemory(ReplayMemory):
    """N-step replay memory"""

    def __init__(self, *args, gamma: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.assert_reserved(keys=['return'])

        self.data['return'] = np.zeros_like(self.data['reward'])
        self.gamma = gamma
        self.last_index = 0

    def end_trajectory(self):
        index = self.index - 1

        # compute n-step returns:
        if self.index > self.last_index:
            rewards = self.data['reward'][self.last_index:index]
        else:
            rewards = np.concatenate([self.data['reward'][self.last_index:],
                                      self.data['reward'][:index]])

        returns = utils.rewards_to_go(rewards, discount=self.gamma)

        if self.index > self.last_index:
            self.data['return'][self.last_index:index] = returns
        else:
            self.data['return'][self.last_index:] = returns[:self.current_size - self.last_index]
            self.data['return'][:index] = returns[index:]

        self.last_index = index


# TODO: use "sum-tree structure" when memory size is large (> 4/8/16k ?)
#  https://adventuresinmachinelearning.com/sumtree-introduction-python/
class PrioritizedMemory(NStepMemory):
    """Prioritized Experience Replay (PER) memory, proportional-variant
        - Based on https://github.com/BY571/Soft-Actor-Critic-and-Extensions/blob/master/SAC_PER.py
    """

    def __init__(self, *args, alpha: DynamicParameter, beta: DynamicParameter, **kwargs):
        """Arguments:
            - alpha: if 0 => uniform sampling, 1 => prioritized sampling.
            - beta: if 0 => no IS-weights correction, 1 => full bias correction.
        """
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.beta = beta
        self.priorities = np.ones(shape=self.size, dtype=np.float64)

        self.indices = None
        self.weights = None
        self.td_error = None

    def store(self, transition: dict):
        super().store(transition)
        self.priorities[self.index - 1] = self.priorities[:self.index].max()

    def sample(self, batch_size: int) -> dict:
        """Samples a batch of transitions"""
        batch = dict()
        size = self.current_size

        # init td-error here (since `batch_size` may vary - should not!)
        # Also, td-error is a tf.Variable in order to deal with SymbolicTensor annoyances
        self.td_error = self._get_var(batch_size)

        # sample transitions according to priority
        probs = self.priorities[:size] ** float(self.alpha())
        probs /= probs.sum()
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        indices = self.random.choice(size, size=batch_size, replace=False, p=probs)
        self.indices = indices

        for key, value in self.data.items():
            batch[key] = self._gather(value, indices)

        # Importance-sampling weights
        weights = (size * probs[indices]) ** float(-self.beta())
        weights /= weights.max()

        if '_weights' in batch:
            raise ValueError('Key "_weights" is reserved for `PrioritizedMemory`.')

        batch['_weights'] = tf.expand_dims(tf.cast(weights, dtype=tf.float32), axis=-1)
        return batch

    def on_update(self):
        self.update_priorities()

    @staticmethod
    def _get_var(size) -> tf.Variable:
        return tf.Variable(tf.zeros(size,), trainable=False, dtype=tf.float32)

    def update_priorities(self, eps=utils.NP_EPS):
        # adding an `eps` ensures probabilities to be non-zero
        self.priorities[self.indices] = np.abs(self.td_error.value()) + eps


# TODO: test
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

    def sample(self, batch_size: int) -> dict:
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

        # indices = tf.random.shuffle(indices, seed=self.seed)[:batch_size]
        indices = self.random.choice(indices, size=batch_size)

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
