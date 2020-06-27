import numpy as np

from functools import partial


# TODO: use numpy instead of python's list, it's faster!
class Memory(object):
    def __init__(self, capacity: int):
        assert capacity > 0
        self.buffer = []
        self.size = 0
        self.capacity = capacity

        # TODO: al posto di 'buffer' mettere un campo per actions, rewards, states, n_states, terminals

    def append(self, observation):
        if len(self.buffer) == self.capacity:
            raise ValueError(f"Maximum capacity ({self.capacity}) reached.")

        self.buffer.append(observation)
        self.size += 1

    # TODO: retrieve should also remove the retrieved elements? (add as parameter?)
    def retrieve(self, amount: int):
        raise NotImplementedError

    def clear(self):
        self.size = 0
        self.buffer.clear()


class Recent(Memory):
    """Recent memory"""

    def retrieve(self, amount: int) -> list:
        assert isinstance(amount, int) and amount > 0
        assert amount <= self.size

        # retrieve last amount elements
        return self.buffer[-amount:self.size]


class Replay(Memory):
    """Replay memory"""
    def __init__(self, *args, allow_repetitions=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.np_random_choice = partial(np.random.choice, replace=allow_repetitions)

    def retrieve(self, amount: int):
        assert isinstance(amount, int) and amount > 0
        # assert amount <= self.size

        # retrieves amount random observations
        return self.np_random_choice(self.buffer, size=amount)
