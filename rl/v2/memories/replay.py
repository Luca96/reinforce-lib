import numpy as np
import tensorflow as tf

from rl.v2.memories import Memory, TransitionSpec
from rl.v2.memories.episodic import EpisodicMemory

from typing import Union


class ReplayMemory(Memory):

    def __init__(self, transition_spec: TransitionSpec, size: int):
        assert size >= 1
        super().__init__(transition_spec)

        self.max_size = int(size)

    def get_batch(self, batch_size: int, **kwargs) -> dict:
        return self.sample(batch_size, **kwargs)

    def sample(self, batch_size: int, seed=None) -> dict:
        """Samples a batch of transitions"""
        batch = dict()

        # random indices
        indices = tf.range(start=0, limit=self.size, dtype=tf.int32)
        indices = tf.random.shuffle(indices, seed=seed)[:batch_size]

        for key, tensor in self.data.items():
            batch[key] = self._get(tensor, indices)

        return batch

    def _get(self, tensor: Union[tf.Tensor, dict], indices: tf.Tensor) -> Union[tf.Tensor, dict]:
        if not isinstance(tensor, dict):
            return tf.gather(tensor, indices)
        else:
            tensors = dict()

            for k, v in tensor.items():
                tensors[k] = self._get(tensor=v, indices=indices)

            return tensors

    def ensure_size(self):
        """Removes oldest transitions to ensure its maximum size is respected"""
        elements_to_remove = self.size - self.max_size

        if elements_to_remove <= 0:
            return

        def ensure(data, key_, tensor_):
            if tf.is_tensor(tensor):
                data[key] = tensor_[elements_to_remove:]
            else:
                for k, v in tensor_.items():
                    ensure(data=data[key_], key_=k, tensor_=v)

        for key, tensor in self.data.items():
            ensure(self.data, key, tensor)

        self.size = self.max_size


class Replay2(ReplayMemory):

    def __init__(self, transition_spec: TransitionSpec, size: int):
        self.data = dict()
        self.size = int(size)
        self.index = 0
        self.specs = transition_spec
        self.full = False

        for name, spec in self.specs.items():
            self.data[name] = self._add_spec(spec)

    def _add_spec(self, spec: dict):
        if 'shape' in spec:
            shape = spec['shape']
            return np.zeros(shape=(self.size,) + shape[1:], dtype=np.float32)  # spec['dtype']

    def store(self, transition: dict):
        """Stores one transition"""
        if self.index >= self.size:
            self.full = True
            self.index = 0

        for k, v in transition.items():
            if k not in self.specs:
                continue

            self._store(data=self.data, spec=self.specs[k], key=k, value=v)

        self.index += 1

    def _store(self, data, spec, key, value):
        if not isinstance(value, dict):
            tensor = tf.cast(value, dtype=spec['dtype'])
            tensor = tf.reshape(tensor, shape=spec['shape'])

            data[key][self.index] = tensor
        else:
            for k, v in value.items():
                self._store(data=data[key], spec=spec[k], key=k, value=v)

    def ensure_size(self):
        pass
