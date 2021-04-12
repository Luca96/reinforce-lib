"""Deep Q-Learning (DQN) with Experience Replay"""

import os
import gym
import numpy as np
import tensorflow as tf

from typing import Union, List, Dict, Tuple

from rl import utils
from rl.parameters import DynamicParameter

from rl.v2.agents import Agent
from rl.v2.memories import TransitionSpec, ReplayMemory, PrioritizedMemory
from rl.v2.networks.q import Network, QNetwork, DoubleQNetwork


class DQN(Agent):
    # TODO: n_step (horizon) updates
    def __init__(self, *args, name='dqn-agent', lr: utils.DynamicType = 3e-4, optimizer='adam', memory_size=1024,
                 policy='e-greedy', epsilon: utils.DynamicType = 0.05, clip_norm: utils.DynamicType = 1.0, load=False,
                 update_target_network: Union[bool, int] = False, polyak: utils.DynamicType = 0.995, double=True,
                 network: dict = None, dueling=True, horizon=1, prioritized=False, alpha: utils.DynamicType = 0.6,
                 beta: utils.DynamicType = 0.4, **kwargs):
        assert horizon >= 1
        assert policy.lower() in ['boltzmann', 'softmax', 'e-greedy', 'greedy']
        super().__init__(*args, name=name, **kwargs)

        self.memory_size = memory_size
        self.policy = policy.lower()
        self.epsilon = DynamicParameter.create(value=epsilon)
        self.polyak = DynamicParameter.create(value=polyak)
        self.prioritized = bool(prioritized)
        self.horizon = int(horizon)

        # PER memory params:
        if self.prioritized:
            self.alpha = DynamicParameter.create(value=alpha)
            self.beta = DynamicParameter.create(value=beta)

        if not update_target_network and self.polyak.value == 1.0:
            self.should_update_target = False
        else:
            self.should_update_target = True
            self.update_target_freq = int(update_target_network)
            self._update_target_freq = self.update_target_freq  # copy

        self.lr = DynamicParameter.create(value=lr)

        self.weights_path = dict(dqn=os.path.join(self.base_path, 'dqn'))

        if double:
            self.dqn = Network.create(agent=self, dueling=dueling, prioritized=self.prioritized, **(network or {}),
                                      base_class=DoubleQNetwork)
        else:
            self.dqn = Network.create(agent=self, dueling=dueling, prioritized=self.prioritized, **(network or {}),
                                      base_class=QNetwork)

        self.dqn.compile(optimizer, clip_norm=clip_norm, learning_rate=self.lr)

        if load:
            self.load()

    @property
    def transition_spec(self) -> TransitionSpec:
        return TransitionSpec(state=self.state_spec, next_state=True, action=(self.num_actions,))

    @property
    def memory(self) -> Union[ReplayMemory, PrioritizedMemory]:
        if self._memory is None:
            if self.prioritized:
                self._memory = PrioritizedMemory(self.transition_spec, shape=self.memory_size, alpha=self.alpha,
                                                 beta=self.beta, seed=self.seed)
            else:
                self._memory = ReplayMemory(self.transition_spec, shape=self.memory_size, seed=self.seed)

        return self._memory

    # TODO: add `MultiDiscrete` action-space support
    def _init_action_space(self):
        assert isinstance(self.env.action_space, gym.spaces.Discrete)

        self.num_actions = 1
        self.num_classes = self.env.action_space.n

        self.convert_action = lambda a: tf.cast(tf.squeeze(a), dtype=tf.int32).numpy()

    # TODO: implement explorative policies directly into dqn.act?
    def act(self, state):
        other, debug = None, {}

        if self.policy == 'boltzmann':
            q_values = self.dqn.q_values(state)
            exp_q = tf.exp(tf.minimum(q_values, 80.0))  # clip to prevent tf.exp go to "inf"
            action = tf.random.categorical(logits=exp_q, num_samples=1, seed=self.seed)
            debug['exp_q_values'] = exp_q

        elif self.policy == 'softmax':
            q_values = self.dqn.q_values(state)
            action = tf.random.categorical(logits=q_values, num_samples=1, seed=self.seed)

        elif self.policy == 'e-greedy':
            eps = self.epsilon()

            # compute probabilities for best action (a* = argmax Q(s,a)), and other actions (a != a*)
            prob_other = tf.cast(eps / self.num_classes, dtype=tf.float32)
            prob_best = 1.0 - eps + prob_other

            q_values = self.dqn.q_values(state)
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
        with utils.Timed('Learn'):
            super().learn(*args, **kwargs)

    def update(self):
        if not self.memory.full_enough(amount=self.batch_size):
            return self.memory.update_warning(self.batch_size)

        with utils.Timed('Update', silent=True):
            batch = self.memory.get_batch(batch_size=self.batch_size)
            self.dqn.train_step(batch)

            if self.prioritized:
                self.memory.update_priorities()

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

        if (timestep % self.horizon == 0) or transition['terminal'] or (timestep == self.max_timesteps):
            self.update()
            self.update_target_network()

    def save_weights(self):
        self.dqn.save_weights(filepath=self.weights_path['dqn'])

    def load_weights(self):
        self.dqn.load_weights(filepath=self.weights_path['dqn'], by_name=False)

    def summary(self):
        self.dqn.summary()


if __name__ == '__main__':
    from rl.parameters import StepDecay, ExponentialDecay

    agent = DQN(env='CartPole-v0', batch_size=32, policy='boltzmann', memory_size=4096,
                name='dqn-cart', clip_norm=5.0, network=dict(num_layers=3),
                lr=StepDecay(1.5e-5, steps=250, rate=0.5),
                reward_scale=0.5, prioritized=True, horizon=4,
                use_summary=False, double=True, dueling=True, seed=42)
    agent.summary()

    agent.learn(episodes=500, timesteps=200//2, render=10, exploration_steps=4096 * 0,
                evaluation=dict(episodes=50, freq=100))
