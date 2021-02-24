import tensorflow as tf

from rl.v2.memories import Memory

from typing import Union


class ReplayMemory(Memory):

    def get_batch(self, batch_size: int, **kwargs) -> dict:
        return self.sample(batch_size, **kwargs)

    def sample(self, batch_size: int, seed=None) -> dict:
        """Samples a batch of transitions"""
        batch = dict()

        # random indices
        indices = tf.range(start=0, limit=self.size, dtype=tf.int32)
        indices = tf.random.shuffle(indices, seed=seed)[:batch_size]

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
