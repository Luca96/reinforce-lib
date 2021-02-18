
import tensorflow as tf

from typing import Union, List, Tuple, Dict


class TransitionSpec:
    """Defines the structure of a transition"""

    def __init__(self, state, action, reward: Union[dict, bool, None] = True, terminal: Union[dict, bool, None] = True,
                 other: dict = None, info: dict = None, next_state=False):
        self.specs = dict()

        if state is not None:
            if list(state.keys()) == ['state']:
                state = state['state']

            self.specs['state'] = self.get_spec(spec=state)

            if next_state:
                self.specs['next_state'] = self.specs['state']

        if action is not None:
            self.specs['action'] = self.get_spec(spec=action)

        if reward is True:
            self.specs['reward'] = dict(shape=(1, 1), dtype=tf.float32)

        elif (reward is not None) and (reward is not False):
            self.specs['reward'] = self.get_spec(spec=reward)

        # if discount is not None:
        #     self.specs['discount'] = self.get_spec(spec=discount)

        if terminal is True:
            self.specs['terminal'] = dict(shape=(1, 1), dtype=tf.float32)

        elif (terminal is not None) and (terminal is not False):
            self.specs['terminal'] = self.get_spec(spec=terminal)

        if other is not None:
            # self.specs['other'] = self.get_spec(spec=other)
            for key, spec in other.items():
                if key in self.specs:
                    raise ValueError(f'Key "other[{key}]" already in use! Try using "other_{key}", instead.')

                self.specs[key] = self.get_spec(spec=spec)

        if info is not None:
            # self.specs['info'] = self.get_spec(spec=info)
            for key, spec in other.items():
                if key in self.specs:
                    raise ValueError(f'Key "info[{key}]" already in use! Try using "info_{key}", instead.')

                self.specs[key] = self.get_spec(spec=spec)

    def __getitem__(self, key) -> dict:
        return self.specs[key]

    def __contains__(self, key):
        return self.specs.__contains__(key)

    def get_spec(self, spec) -> dict:
        if isinstance(spec, tuple):
            return dict(shape=(1,) + spec, dtype=tf.float32)

        elif isinstance(spec, dict):
            if 'shape' in spec:
                assert isinstance(spec['shape'], tuple)
                return dict(shape=(1,) + spec['shape'], dtype=spec.get('dtype', tf.float32))
            else:
                new_spec = dict()

                for k, v in spec.items():
                    new_spec[k] = self.get_spec(spec=v)

                return new_spec
        else:
            raise ValueError(f"`Spec` should be one of 'dict' or 'tuple' not '{type(spec)}'.")

    def keys(self):
        return self.specs.keys()

    def values(self):
        return self.specs.values()

    def items(self):
        return self.specs.items()

    # def add_spec(self, name: str, spec):
    #     if spec is None:
    #         return
    #         # raise ValueError(f'Spec "{prefix}_{name}" = None, was found. Remove or define it.')
    #
    #     if isinstance(spec, tuple):
    #         self._check_spec(name)
    #         self.specs[name] = dict(shape=(1,) + spec, dtype=tf.float32)
    #
    #     elif isinstance(spec, dict):
    #         if 'shape' in spec:
    #             self._check_spec(name)
    #             self.specs[name] = dict(shape=(1,) + spec['shape'], dtype=spec.get('dtype', tf.float32))
    #         else:
    #             for k, v in spec.items():
    #                 self.add_spec(name=f'{name}_{k}', spec=v)
    #     else:
    #         raise ValueError(f"`Spec` should be one of 'dict', 'tuple', or 'None' not '{type(spec)}'.")
    #
    # def _check_spec(self, name):
    #     if name in self.specs:
    #         raise ValueError(f'A spec named "{name}" was already added. Please use another name.')
