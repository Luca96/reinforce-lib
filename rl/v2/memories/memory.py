
import gc
import tensorflow as tf

from rl.v2.memories import TransitionSpec

from typing import Union


# TODO: use tf.Variable as data-type that also supports assignment...
# TODO: consider to allocate the entire memory
# TODO: use numpy's arrays or utils.DynamicArray, instead of `tf.tensor + tf.concat`
class Memory:
    """Base class for Memories (e.g. recent, replay, prioritized, ...)"""

    def __init__(self, transition_spec: TransitionSpec):
        self.data = dict()
        self.size = 0
        self.specs = transition_spec

        for name, spec in self.specs.items():
            self.data[name] = self._add_spec(spec)

    def _add_spec(self, spec: dict):
        if 'shape' in spec:
            shape = spec['shape']
            return tf.zeros(shape=(0,) + shape[1:], dtype=spec['dtype'])

        # spec is a dict with more specs inside (so, recurse)
        data = dict()

        for k, v in spec.items():
            data[k] = self._add_spec(spec=v)

        return data

    def full_enough(self, amount: int) -> bool:
        """Tests whether the memory contains at least `amount` elements."""
        return self.size >= amount

    def store(self, transition: dict):
        """Stores one transition"""
        size = 0

        for k, v in transition.items():
            if k not in self.specs:
                continue

            self.data[k], size = self._store(data=self.data, spec=self.specs[k], key=k, value=v)

        self.size = size

    def _store(self, data, spec, key, value):
        if not isinstance(value, dict):
            tensor = tf.cast(value, dtype=spec['dtype'])
            tensor = tf.reshape(tensor, shape=spec['shape'])
            tensor = tf.concat([data[key], tensor], axis=0)

            return tensor, int(tensor.shape[0])
        else:
            tensors = dict()
            size = 0

            for k, v in value.items():
                tensors[k], size = self._store(data=data[key], spec=spec[k], key=k, value=v)

            return tensors, size

    def get_batch(self, batch_size: int, **kwargs) -> dict:
        raise NotImplementedError

    def get_batches(self, amount: int, batch_size: int, **kwargs):
        assert amount >= 1

        for _ in range(amount):
            yield self.get_batch(batch_size, **kwargs)

    def to_batches(self, batch_size: int, **kwargs):
        """Returns a tf.data.Dataset iterator over batches of transitions"""
        raise NotImplementedError

    def clear(self):
        """Empties the memory"""
        for key, tensor in self.data.items():
            self._clear(self.data, key, tensor)

        self.size = 0
        gc.collect()

    def _clear(self, data, key, tensor: Union[tf.Tensor, dict]):
        if tf.is_tensor(tensor):
            data[key] = tensor[:0]  # remove all elements
        else:
            # tensor is dict
            for k, v in tensor.items():
                self._clear(data=data[key], key=k, tensor=v)

    def __delete__(self, instance):
        pass

    def serialize(self, path: str):
        """Saves the entire content of the memory into a numpy's npz file"""
        pass

    @staticmethod
    def deserialize(path: str) -> 'Memory':
        """Creates a Memory instance from a numpy's nps file"""
        pass
