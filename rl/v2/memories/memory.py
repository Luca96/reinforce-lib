
import numpy as np

from rl import utils
from rl.v2.memories import TransitionSpec

from typing import Union, Tuple


class Memory:
    """A circular buffer that supports uniform replying"""

    def __init__(self, transition_spec: TransitionSpec, shape: Union[int, Tuple]):
        if isinstance(shape, tuple):
            self.size = np.prod(shape)
            self.shape = shape

        elif isinstance(shape, (int, float)):
            self.shape = (int(shape),)
            self.size = self.shape[0]
        else:
            raise ValueError(f'Argument "shape" must be a `tuple`, `int`, or `float` not {type(shape)}.')

        assert self.size >= 1

        self.data = dict()
        self.index = 0
        self.specs = transition_spec
        self.full = False

        for name, spec in self.specs.items():
            self.data[name] = self._add_spec(spec)

    # TODO: should be recursive?
    def _add_spec(self, spec: dict):
        if 'shape' in spec:
            shape = spec['shape']
            # return np.zeros(shape=(self.size,) + shape, dtype=spec['dtype'])
            return np.zeros(shape=self.shape + shape, dtype=spec['dtype'])

    def is_full(self) -> bool:
        if self.full:
            return True

        if self.index >= self.size:
            self.full = True

        return self.full

    def full_enough(self, amount: int) -> bool:
        """Tests whether the memory contains at least `amount` elements."""
        assert 0 < amount <= self.size
        return self.full or self.index >= amount

    def store(self, transition: dict):
        """Stores one transition"""
        if self.is_full():
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

    def to_batches(self, batch_size: int, repeat=0, **kwargs):
        """Returns a tf.data.Dataset iterator over batches of transitions"""
        batches = utils.data_to_batches(tensors=self.get_data(), batch_size=batch_size, **kwargs)

        if repeat > 0:
            return batches.repeat(count=repeat)

        return batches

    def get_data(self) -> dict:
        """Returns the whole data in memory as a single batch"""
        if self.full:
            return self.data

        def _get(data, k_, val):
            if not isinstance(val, dict):
                data[k_] = val[:self.index]
            else:
                data[k_] = dict()

                for k, v in val.items():
                    _get(data[k_], k, v)

        all_data = dict()

        for key, value in self.data.items():
            _get(all_data, key, value)

        return all_data

    def clear(self):
        """Empties the memory"""
        self.index = 0
        self.full = False

    def __delete__(self, instance):
        pass

    def summary(self):
        """Summarizes the structure of the current memory"""
        print('-' * 80)
        print(f'Memory: "{self.__class__.__name__}"')
        print('-' * 80)

        def _summary(key, value, nesting=0):
            if isinstance(value, dict):
                print('  ' * nesting + f' - {key}:')

                for k_, v_ in value.items():
                    _summary(key=k_, value=v_, nesting=nesting + 1)
            else:
                print('  ' * nesting + f' - {key}: shape {value.shape}, dtype {value.dtype}')
                print('-' * 80)

        for k, v in self.data.items():
            _summary(key=k, value=v, nesting=0)

    def serialize(self, path: str):
        """Saves the entire content of the memory into a numpy's npz file"""
        pass

    @staticmethod
    def deserialize(path: str) -> 'Memory':
        """Creates a Memory instance from a numpy's nps file"""
        pass
