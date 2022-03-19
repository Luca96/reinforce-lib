"""Twin-delayed DDPG (TD3)"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import gym
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Layer, Dense, Concatenate
from typing import Dict, List

from rl import utils
from rl.parameters import DynamicParameter

from rl.v2.agents import Agent
from rl.v2.agents.ddpg import DDPG, Network, CriticNetwork, ActorNetwork
from rl.v2.networks import backbones


class TwinCriticNetwork(CriticNetwork):

    def __init__(self, *args, log_prefix='twin-q', **kwargs):
        self.init_hack()

        self.q1: tf.keras.Model = None
        self.q2: tf.keras.Model = None

        super().__init__(*args, log_prefix=log_prefix, **kwargs)

    # TODO: avoid always calling both networks
    def call(self, *inputs, both_q=False, training=None, **kwargs):
        q1, q2 = super().call(*inputs, training=training, **kwargs)

        if both_q:
            return q1, q2

        return q1

    def structure(self, inputs: Dict[str, Input], name='TD3-TwinCriticNetwork', **kwargs) -> tuple:
        self.output_kwargs['name'] = 'q1-values'
        _, output1, name1 = super().structure(inputs, name=name + '-Q1', **kwargs)

        self.output_kwargs['name'] = 'q2-values'
        inputs, output2, name2 = super().structure(inputs, name=name + '-Q2', **kwargs)

        # create two networks
        self.q1 = tf.keras.Model(inputs, outputs=output1, name=name1)
        self.q2 = tf.keras.Model(inputs, outputs=output2, name=name2)

        # return joint model
        return inputs, (self.q1.output, self.q2.output), name

    def output_layer(self, layer: Layer, **kwargs) -> Layer:
        return Dense(units=self.agent.num_actions, **self.output_kwargs)(layer)

    @tf.function
    def objective(self, batch: dict, reduction=tf.reduce_mean) -> tuple:
        q_values1, q_values2 = self((batch['state'], batch['action']), both_q=True, training=True)
        q_targets, debug = self.targets(batch)

        # loss is the sum of both TD-errors
        loss_q1 = reduction(0.5 * tf.square(q_values1 - q_targets))
        loss_q2 = reduction(0.5 * tf.square(q_values2 - q_targets))
        loss = loss_q1 + loss_q2

        debug.update(loss=loss, loss_q1=loss_q1, loss_q2=loss_q2, q_values1=q_values1, q_values2=q_values2)
        return loss, debug

    @tf.function
    def targets(self, batch: dict):
        next_states = batch['next_state']

        clipped_noise = self._clipped_noise(shape=batch['action'].shape)
        actions = self.agent.actor.target(next_states, training=False)

        noisy_actions = actions + clipped_noise
        noisy_actions = tf.clip_by_value(noisy_actions, clip_value_min=-1.0, clip_value_max=1.0)

        q_values1, q_values2 = self.target((next_states, noisy_actions), both_q=True, training=False)
        min_q_values = tf.minimum(q_values1, q_values2)

        targets = batch['reward'] + self.agent.gamma * min_q_values * (1.0 - batch['terminal'])
        targets = tf.stop_gradient(targets)

        return targets, dict(noise_clipped=clipped_noise, noisy_actions=noisy_actions, targets=targets,
                             min_q_values=min_q_values, next_q_values1=q_values1, next_q_values2=q_values2)

    def _clipped_noise(self, shape: tuple) -> tf.Tensor:
        noise = tf.random.normal(shape, stddev=self.agent.noise.value, seed=self.agent.seed)
        return tf.clip_by_value(noise, clip_value_min=-self.agent.clip_noise, clip_value_max=self.agent.clip_noise)


class TD3(DDPG):
    def __init__(self, *args, name='td3', clip_noise=0.5, actor_update_freq=2, **kwargs):
        assert clip_noise >= 0.0
        assert actor_update_freq >= 1

        self.clip_noise = float(clip_noise)
        self.actor_update_freq = int(actor_update_freq)

        # switch `CriticNetwork` for `TwinCriticNetwork`
        critic = kwargs.pop('critic', {})
        critic.setdefault('class', TwinCriticNetwork)

        super().__init__(*args, name=name, critic=critic, **kwargs)

    def update(self):
        batch = self.memory.get_batch(batch_size=self.batch_size)

        self.critic.train_step(batch)

        if self.total_steps % self.actor_update_freq == 0:
            self.actor.train_step(batch)
            self.update_target_networks()


if __name__ == '__main__':
    utils.set_random_seed(42)

    # agent = TD3(env='Pendulum-v1', actor_lr=1e-3, critic_lr=1e-3, polyak=0.995, actor_update_freq=2,
    #             memory_size=100_000, batch_size=256, name='td3-pendulum', use_summary=True,
    #             actor=dict(units=256), critic=dict(units=256), noise=0.1)

    # agent = TD3(env='Pendulum-v1', actor_lr=1e-3, critic_lr=1e-3, polyak=0.995, actor_update_freq=2,
    #             memory_size=256_000, batch_size=128, name='td3-pendulum', use_summary=True,
    #             actor=dict(units=64), critic=dict(units=64), noise=0.1, seed=utils.GLOBAL_SEED)
    #
    # # agent.summary()
    # # breakpoint()
    #
    # # fix specific to pendulum env
    # agent._convert_action = agent.convert_action
    # agent.convert_action = lambda a: np.reshape(agent._convert_action(a), newshape=[1])
    #
    # agent.learn(episodes=200, timesteps=200, evaluation=dict(freq=10, episodes=20),
    #             exploration_steps=5 * agent.batch_size, save=True)

    # agent = TD3(env='LunarLanderContinuous-v2', actor_lr=1e-3, critic_lr=1e-3, polyak=0.995, actor_update_freq=2,
    #             memory_size=256_000, batch_size=128, name='td3-lunar', use_summary=True,
    #             actor=dict(units=64), critic=dict(units=64), noise=0.1, seed=utils.GLOBAL_SEED)
    #
    # agent.learn(episodes=250, timesteps=200, evaluation=dict(freq=10, episodes=20),
    #             exploration_steps=5 * agent.batch_size, save=True)
    # exit()

    # # solved when r = 300 in 1600 timesteps
    # agent = TD3(env='BipedalWalker-v3', actor_lr=1e-3, critic_lr=1e-3, polyak=0.999, actor_update_freq=2,
    #             memory_size=256_000, batch_size=256, name='td3-walker', use_summary=True, seed=42,
    #             actor=dict(units=256), critic=dict(units=256), noise=0.1)
    #
    # agent.learn(episodes=2500, timesteps=1600, evaluation=dict(freq=20, episodes=20),
    #             save=True, exploration_steps=5 * agent.batch_size)
    # exit()

    import pybullet_envs

    # 2300+ reward at episode 4690

    agent = TD3(env='HopperBulletEnv-v0', name='td3-hopper',
                use_summary=True, seed=42,
                actor_lr=1e-3, critic_lr=1e-3,
                polyak=0.999, actor_update_freq=2,
                memory_size=256_000, batch_size=256,
                actor=dict(units=256), critic=dict(units=256), noise=0.1)

    agent.learn(episodes=5_000, timesteps=1000, evaluation=dict(freq=20, episodes=20),
                save=True, exploration_steps=5 * agent.batch_size)

    agent.record(timesteps=1000)
