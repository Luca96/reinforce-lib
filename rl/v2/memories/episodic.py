
import numpy as np
import tensorflow as tf

from rl import utils
from rl.v2.memories import Memory


class EpisodicMemory(Memory):

    def get_data(self, pad=False) -> dict:
        if self.full:
            index = self.size
        else:
            index = self.index

        def _get(_data, _k, val):
            if not isinstance(val, dict):
                if pad:
                    _data[_k] = val
                else:
                    _data[_k] = val[:index]
            else:
                _data[_k] = dict()

                for k, v in val.items():
                    _get(_data[_k], k, v)

        data = dict()

        for key, value in self.data.items():
            _get(data, key, value)

        return data


class GAEMemory(EpisodicMemory):
    """Episodic memory with Generalized Advantage Estimation (GAE)"""

    def __init__(self, *args, agent, **kwargs):
        super().__init__(*args, **kwargs)

        if 'return' in self.data:
            raise ValueError('Key "return" is reserved!')

        if 'advantage' in self.data:
            raise ValueError('Key "advantage" is reserved!')

        self.data['return'] = np.zeros_like(self.data['value'])
        self.data['advantage'] = np.zeros(shape=self.shape + (1,), dtype=np.float32)
        self.agent = agent

    def end_trajectory(self, last_value) -> dict:
        data_reward, data_value = self.data['reward'], self.data['value']
        data_return, data_adv = self.data['return'], self.data['advantage']

        v = tf.reshape(last_value, shape=(1, -1))
        rewards = np.concatenate([data_reward[:self.index], v], axis=0)
        values = np.concatenate([data_value[:self.index], v], axis=0)

        # compute returns and advantages
        returns, ret_norm = self.compute_returns(rewards)
        adv, adv_norm = self.compute_advantages(rewards, values)

        # store them
        data_return[:self.index] = ret_norm
        data_adv[:self.index] = adv_norm

        return dict(returns=returns, returns_normalized=ret_norm, advantages=adv, advantages_normalized=adv_norm,
                    values=values, advantages_hist=adv_norm, returns_hist=ret_norm)

    def compute_returns(self, rewards):
        returns = utils.rewards_to_go(rewards, discount=self.agent.gamma)
        return returns, self.agent.returns_norm_fn(returns)

    def compute_advantages(self, rewards, values):
        advantages = utils.gae(rewards, values=values, gamma=self.agent.gamma, lambda_=self.agent.lambda_)
        adv_norm = self.agent.adv_normalization_fn(advantages) * self.agent.adv_scale()
        return advantages, adv_norm
