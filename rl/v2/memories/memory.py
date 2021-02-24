
import numpy as np

from rl import utils
from rl.v2.memories import TransitionSpec


class Memory:
    """A circular buffer that supports uniform replying"""

    def __init__(self, transition_spec: TransitionSpec, size: int):
        assert size >= 1

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
            return np.zeros(shape=(self.size,) + shape, dtype=spec['dtype'])

    def full_enough(self, amount: int) -> bool:
        """Tests whether the memory contains at least `amount` elements."""
        assert 0 < amount <= self.size
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
            array = np.reshape(value, newshape=spec['shape'])

            data[key][self.index] = array
        else:
            for k, v in value.items():
                if k not in spec:
                    continue

                self._store(data=data[key], spec=spec[k], key=k, value=v)

    def get_batch(self, batch_size: int, **kwargs) -> dict:
        raise NotImplementedError

    def get_batches(self, amount: int, batch_size: int, **kwargs):
        assert amount >= 1

        for _ in range(amount):
            yield self.get_batch(batch_size, **kwargs)

    def to_batches(self, batch_size: int, **kwargs):
        """Returns a tf.data.Dataset iterator over batches of transitions"""
        if self.full:
            return utils.data_to_batches(tensors=self.data, batch_size=batch_size, **kwargs)

        return utils.data_to_batches(tensors=self.data, batch_size=batch_size, take=self.index, **kwargs)

    def get_data(self) -> dict:
        """Returns the whole data in memory as a single batch"""
        pass

    def clear(self):
        """Empties the memory"""
        self.index = 0
        self.full = False

    def __delete__(self, instance):
        pass

    def serialize(self, path: str):
        """Saves the entire content of the memory into a numpy's npz file"""
        pass

    @staticmethod
    def deserialize(path: str) -> 'Memory':
        """Creates a Memory instance from a numpy's nps file"""
        pass
