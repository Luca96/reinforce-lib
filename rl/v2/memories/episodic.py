
from rl.v2.memories import Memory


class EpisodicMemory(Memory):

    def get_data(self) -> dict:
        if self.full:
            index = self.size
        else:
            index = self.index

        def _get(_data, _k, val):
            if not isinstance(val, dict):
                _data[_k] = val[:index]
            else:
                _data[_k] = dict()

                for k, v in val.items():
                    _get(_data[_k], k, v)

        data = dict()

        for key, value in self.data.items():
            _get(data, key, value)

        return data
