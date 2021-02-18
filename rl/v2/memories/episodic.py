
import tensorflow as tf

from rl.v2.memories import Memory, TransitionSpec
from rl import utils


class EpisodicMemory(Memory):
    """Episodic memory is a circular buffer"""

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

            return tf.Variable(initial_value=tf.zeros(shape=(self.size,) + shape[1:]),
                               dtype=spec['dtype'], trainable=False)

        # spec is a dict with more specs inside (so, recurse)
        data = dict()

        for k, v in spec.items():
            data[k] = self._add_spec(spec=v)

        return data

    def full_enough(self, amount: int) -> bool:
        return self.full or self.index >= amount

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
            tensor = tf.reshape(tensor, shape=spec['shape'][1:])

            data[key][self.index].assign(tensor)
        else:
            for k, v in value.items():
                self._store(data=data[key], spec=spec[k], key=k, value=v)

    def get_data(self) -> dict:
        if self.full:
            index = self.size
        else:
            index = self.index

        def _get(_data, _k, _v):
            if not isinstance(_v, dict):
                _data[_k] = _v[:index]
            else:
                _data[_k] = dict()

                for k, v in _v.items():
                    _get(_data[_k], k, v)

        data = dict()

        for key, value in self.data.items():
            _get(data, key, value)

        return data

    def to_batches(self, batch_size: int, **kwargs):
        if self.full:
            return utils.data_to_batches(tensors=self.data, batch_size=batch_size, **kwargs)

        return utils.data_to_batches(tensors=self.data, batch_size=batch_size, take=self.index, **kwargs)
