
import tensorflow as tf

from rl.v2.memories import Memory, TransitionSpec

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
