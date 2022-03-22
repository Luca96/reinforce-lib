"""Rainbow Agent (= C51 + PER + Dueling architecture + Double DQN + Noisy Networks + N-step returns)"""

import gym
import numpy as np
import tensorflow as tf

from typing import Union, List, Dict, Tuple

from rl import utils

from rl.v2.agents import DQN
from rl.v2.networks.q import RainbowQNetwork


class Rainbow(DQN):
    # https://github.com/google/dopamine/blob/master/dopamine/agents/rainbow/rainbow_agent.py

    def __init__(self, *args, name='rainbow-agent', lr: utils.DynamicType = 3e-4, optimizer='adam', memory_size=1024,
                 policy='greedy', epsilon: utils.DynamicType = 0.05, clip_norm: utils.DynamicType = None, load=False,
                 update_target_network: Union[bool, int] = False, polyak: utils.DynamicType = 0.995, num_atoms=51,
                 network: dict = None, dueling=True, v_min=-10.0, v_max=10.0, horizon=3, noisy=True, **kwargs):
        assert num_atoms >= 2
        assert v_min < v_max

        self.num_atoms = int(num_atoms)
        self.v_min = tf.constant(v_min, dtype=tf.float32)
        self.v_max = tf.constant(v_max, dtype=tf.float32)

        # support set {z_i}:
        self.support = tf.linspace(self.v_min, self.v_max, num=self.num_atoms)

        network = network or {}
        network.update(noisy=bool(noisy))

        if ('class' not in network) and ('cls' not in network):
            network['class'] = RainbowQNetwork

        super().__init__(*args, name=name, lr=lr, optimizer=optimizer, memory_size=memory_size, policy=policy,
                         epsilon=epsilon, clip_norm=clip_norm, load=load, update_target_network=update_target_network,
                         polyak=polyak, double=False, dueling=dueling, network=network, horizon=horizon, **kwargs)


if __name__ == '__main__':
    from rl.presets import DQNPresets

    presets = DQNPresets.CART_POLE.copy()
    presets.pop('double')

    agent = Rainbow.from_preset(presets, name='rainbow-cart', use_summary=True)
    agent.summary()

    agent.learn(episodes=500, timesteps=200, render=False, exploration_steps=500,
                evaluation=dict(episodes=50, freq=100))
