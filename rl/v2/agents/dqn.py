"""Deep Q-Learning (DQN) with Experience Replay"""

import os
import gym
import time
import numpy as np
import tensorflow as tf

from typing import Union, List, Dict, Tuple

from rl import utils
from rl.parameters import DynamicParameter

from rl.v2.agents import Agent
from rl.v2.memories import ReplayMemory, TransitionSpec
from rl.v2.networks.q import DQNetwork


class DQN(Agent):
    def __init__(self, *args, name='dqn-agent', lr: utils.DynamicType = 3e-4, optimizer='adam', memory_size=1024,
                 policy='e-greedy', epsilon: utils.DynamicType = 0.05, clip_norm: utils.DynamicType = 1.0, load=False,
                 update_target_network: Union[bool, int] = False, polyak: utils.DynamicType = 0.995,
                 network: dict = None, update_on_timestep=True, **kwargs):
        assert policy.lower() in ['boltzmann', 'softmax', 'e-greedy', 'greedy']
        super().__init__(*args, name=name, **kwargs)

        self.memory_size = memory_size
        self.policy = policy.lower()
        self.epsilon = DynamicParameter.create(value=epsilon)
        self.polyak = DynamicParameter.create(value=polyak)
        self.update_on_timestep = bool(update_on_timestep)

        if not update_target_network and self.polyak.value == 1.0:
            self.should_update_target = False
        else:
            self.should_update_target = True
            self.update_target_freq = int(update_target_network)
            self._update_target_freq = self.update_target_freq  # copy

        self._init_action_space()

        self.lr = DynamicParameter.create(value=lr)

        self.weights_path = dict(dqn=os.path.join(self.base_path, 'dqn'))

        self.dqn = DQNetwork(agent=self, **(network or {}))
        self.dqn.compile(optimizer, clip_norm=clip_norm, learning_rate=self.lr)

        if load:
            self.load()

    @property
    def transition_spec(self) -> TransitionSpec:
        return TransitionSpec(state=self.state_spec, next_state=True, action=(self.num_actions,))

    @property
    def memory(self) -> ReplayMemory:
        if self._memory is None:
            self._memory = ReplayMemory(self.transition_spec, size=self.memory_size)

        return self._memory

    def act(self, state):
        other, debug = None, {}

        # TODO: implement explorative policies directly into dqn.act?
        if self.policy == 'boltzmann':
            q_values = self.dqn(state, training=False)
            exp_q = tf.exp(q_values)
            action = tf.random.categorical(logits=exp_q, num_samples=1, seed=self.seed)
            debug['exp_q_values'] = exp_q

        elif self.policy == 'softmax':
            q_values = self.dqn(state, training=False)
            action = tf.random.categorical(logits=q_values, num_samples=1, seed=self.seed)

        elif self.policy == 'e-greedy':
            eps = self.epsilon()

            # compute probabilities for best action (a* = argmax Q(s,a)), and other actions (a != a*)
            prob_other = tf.cast(eps / self.num_classes, dtype=tf.float32)
            prob_best = 1.0 - eps + prob_other

            q_values = self.dqn(state, training=False)
            best_action = tf.squeeze(tf.argmax(q_values, axis=-1))

            prob = np.full_like(q_values, fill_value=prob_other)
            prob[0][best_action] = prob_best

            action = tf.random.categorical(logits=prob, num_samples=1, seed=self.seed)
            debug['prob_best'] = prob_best
            debug['prob_other'] = prob_other
        else:
            # greedy
            action = self.dqn.act(state)

        return action, other, debug

    def act_randomly(self, state) -> Tuple[tf.Tensor, dict, dict]:
        actions = tf.random.categorical(logits=tf.ones(shape=(1, self.num_classes)), num_samples=1, seed=self.seed)
        return actions, {}, {}

    def learn(self, *args, **kwargs):
        t0 = time.time()
        super().learn(*args, **kwargs)
        print(f'Time {round(time.time() - t0, 3)}s.')

    def update(self):
        if not self.memory.full_enough(amount=self.batch_size):
            print('Not updated: memory too small!')
            return

        t0 = time.time()
        batch = self.memory.get_batch(batch_size=self.batch_size, seed=self.seed)

        self.dqn.fit(x=batch, y=None, batch_size=self.batch_size, shuffle=False, verbose=0)

        # print(f'Update in {round(time.time() - t0, 3)}s')

    def _init_action_space(self):
        assert isinstance(self.env.action_space, gym.spaces.Discrete)

        self.num_actions = 1
        self.num_classes = self.env.action_space.n

        self.convert_action = lambda a: tf.cast(tf.squeeze(a), dtype=tf.int32).numpy()

    def update_target_network(self):
        if self.should_update_target:
            self.update_target_freq -= 1

            if self.update_target_freq == 0:
                self.dqn.update_target_network(copy_weights=True)
                self.update_target_freq = self._update_target_freq
            else:
                self.dqn.update_target_network(copy_weights=False, polyak=self.polyak())

    def on_transition(self, transition: dict, timestep: int, episode: int):
        super().on_transition(transition, timestep, episode)

        if self.update_on_timestep:
            self.update()
            self.update_target_network()

    def on_termination(self, last_transition, timestep: int, episode: int):
        super().on_termination(last_transition, timestep, episode)

        if not self.update_on_timestep:
            self.update()
            self.update_target_network()

    def save_weights(self):
        self.dqn.save_weights(filepath=self.weights_path['dqn'])

    def load_weights(self):
        self.dqn.load_weights(filepath=self.weights_path['dqn'], by_name=False)

    def summary(self):
        self.dqn.summary()


if __name__ == '__main__':
    agent = DQN(env='CartPole-v0', batch_size=32, policy='e-greedy', lr=1e-3*2,
                log_mode=False, update_on_timestep=False, seed=42)
    agent.summary()

    agent.learn(episodes=500, timesteps=200, render=False,
                evaluation=dict(episodes=100, freq=500))
