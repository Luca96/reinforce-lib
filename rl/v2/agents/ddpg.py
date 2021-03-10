"""Deep Deterministic Policy Gradient (DDPG) Agent"""

import os
import time
import tensorflow as tf

from tensorflow.keras.layers import Layer, Dense, Input

from rl import utils
from rl.parameters import DynamicParameter

from rl.v2.agents import Agent
from rl.v2.memories import ReplayMemory, TransitionSpec
from rl.v2.networks import DeterministicPolicyNetwork, QNetwork, backbones

from typing import Dict, Tuple


class DDPG(Agent):
    def __init__(self, *args, optimizer='adam', actor_lr: utils.DynamicType = 1e-4, critic_lr: utils.DynamicType = 1e-3,
                 memory_size=1024, name='ddpg-agent', actor: dict = None, critic: dict = None, load=False,
                 clip_norm: utils.DynamicType = None, polyak: utils.DynamicType = 0.999, optimization_steps=1,
                 noise: utils.DynamicType = 0.05, **kwargs):
        assert optimization_steps >= 1

        super().__init__(*args, name=name, **kwargs)

        # Hyper-parameters
        self.memory_size = int(memory_size)
        self.clip_norm = self._init_clip_norm(clip_norm)
        self.polyak = DynamicParameter.create(value=polyak)
        self.optimization_steps = int(optimization_steps)
        self.noise = DynamicParameter.create(value=noise)

        self.critic_lr = DynamicParameter.create(value=critic_lr)
        self.actor_lr = DynamicParameter.create(value=actor_lr)

        # Networks
        self.weights_path = dict(actor=os.path.join(self.base_path, 'actor'),
                                 critic=os.path.join(self.base_path, 'critic'))

        self.actor = ActorNetwork(agent=self, target=True, log_prefix='actor', **(actor or {}))
        self.critic = CriticNetwork(agent=self, log_prefix='critic', **(critic or {}))

        self.actor.compile(optimizer, clip_norm=self.clip_norm[0], learning_rate=self.actor_lr)
        self.critic.compile(optimizer, clip_norm=self.clip_norm[1], learning_rate=self.critic_lr)

        if load:
            self.load()

    @property
    def transition_spec(self) -> TransitionSpec:
        if self.distribution_type == 'categorical':
            action_shape = (self.num_classes,)
        else:
            action_shape = (self.num_actions,)

        return TransitionSpec(state=self.state_spec, next_state=True, action=action_shape)

    @property
    def memory(self) -> ReplayMemory:
        if self._memory is None:
            self._memory = ReplayMemory(self.transition_spec, size=self.memory_size)

        return self._memory

    def _init_clip_norm(self, clip_norm):
        if clip_norm is None:
            return None, None

        if isinstance(clip_norm, float) or isinstance(clip_norm, int):
            return clip_norm, clip_norm

        if isinstance(clip_norm, tuple):
            assert len(clip_norm) > 0

            if len(clip_norm) < 2:
                return clip_norm[0], clip_norm[0]
            else:
                return clip_norm

        raise ValueError(f'Parameter "clip_norm" should be `int`, `float` or `tuple` not {type(clip_norm)}.')

    def _init_action_space(self):
        super()._init_action_space()

        if self.distribution_type == 'categorical':
            def convert_action(logits):
                actions = tf.argmax(logits, axis=-1)
                return tf.cast(tf.squeeze(actions), dtype=tf.int32).numpy()

            self.convert_action = convert_action

        elif self.distribution_type == 'beta':
            action_range = self.action_range
            action_low = self.action_low

            def convert_action(action):
                # action comes from a `tanh` output, so lying within [-1, 1]
                action = (action + 1.0) / 2.0  # transform to [0, 1] (as Beta would do)
                return tf.squeeze(action * action_range + action_low).numpy()

            self.convert_action = convert_action

    @tf.function
    def act(self, state) -> Tuple[tf.Tensor, dict, dict]:
        action = self.actor(state, training=True)

        if self.noise.value > 0.0:
            noise = tf.random.normal(shape=action.shape, stddev=self.noise(), seed=self.seed)
            debug = dict(noise=noise, noise_std=self.noise.value, action_not_noise=action)
        else:
            noise = 0.0
            debug = {}

        # TODO: should clip actions + noise?
        return action + noise, {}, debug

    def update(self):
        if not self.memory.full_enough(amount=self.batch_size):
            print('Not updated: memory not enough full.')

        for _ in range(self.optimization_steps):
            batch = self.memory.sample(batch_size=self.batch_size, seed=self.seed)

            self.critic.train_step(batch)
            self.actor.train_step(batch)

            self.update_target_networks()

    def update_target_networks(self):
        if self.polyak.value < 1.0:
            self.actor.update_target_network(polyak=self.polyak())
            self.critic.update_target_network(polyak=self.polyak.value)

    def learn(self, *args, **kwargs):
        t0 = time.time()
        super().learn(*args, **kwargs)
        print(f'Time {round(time.time() - t0, 3)}s.')

    def on_transition(self, transition: dict, timestep: int, episode: int):
        super().on_transition(transition, timestep, episode)
        self.update()

    def save_weights(self):
        self.actor.save_weights(filepath=self.weights_path['actor'])
        self.critic.save_weights(filepath=self.weights_path['critic'])

    def load_weights(self):
        self.actor.load_weights(filepath=self.weights_path['actor'], by_name=False)
        self.critic.load_weights(filepath=self.weights_path['critic'], by_name=False)

    def summary(self):
        self.actor.summary()
        self.critic.summary()


class ActorNetwork(DeterministicPolicyNetwork):

    @tf.function
    def objective(self, batch: dict, reduction=tf.reduce_mean) -> tuple:
        states = batch['state']
        actions = self(states, training=True)

        q_values = self.agent.critic(states, actions, training=False)
        loss = -reduction(q_values)

        return loss, dict(loss=loss, actions=actions, q_values=q_values)


class CriticNetwork(QNetwork):

    def call(self, *inputs, training=None, **kwargs):
        return super().call(inputs, actions=None, training=training)

    def act(self, inputs):
        raise NotImplementedError

    def structure(self, inputs: Dict[str, Input], name='CriticNetwork', **kwargs) -> tuple:
        state_in = inputs['state']

        if self.agent.distribution_type == 'categorical':
            action_in = Input(shape=(self.agent.num_classes,), name='action', dtype=tf.float32)
        else:
            action_in = Input(shape=(self.agent.num_actions,), name='action', dtype=tf.float32)

        x = backbones.dense_branched(state_in, action_in, **kwargs)

        output = self.output_layer(**self.output_args)(x)
        return (state_in, action_in), output, name

    @tf.function
    def objective(self, batch: dict, reduction=tf.reduce_mean) -> tuple:
        q_values = self(batch['state'], batch['action'], training=True)
        targets, debug = self.targets(batch)

        loss = reduction(tf.square(targets - q_values))
        debug.update(loss=loss, q_values=q_values)

        return loss, debug

    @tf.function
    def targets(self, batch):
        next_actions = self.agent.actor.target(batch['next_state'], training=False)
        next_q_values = self.target(batch['next_state'], next_actions, training=False)

        targets = tf.stop_gradient(batch['reward'] + self.agent.gamma * (1.0 - batch['terminal']) * next_q_values)

        return targets, dict(next_actions=next_actions, next_q_values=next_q_values, targets=targets)

    def output_layer(self, **kwargs) -> Layer:
        if self.agent.distribution_type == 'categorical':
            return Dense(units=self.agent.num_classes, name='q-values', **kwargs)

        return Dense(units=self.agent.num_actions, name='q-values', **kwargs)


if __name__ == '__main__':
    env1 = 'LunarLanderContinuous-v2'
    env2 = 'CartPole-v0'
    a = DDPG(env=env2, batch_size=64, actor=dict(units=64), critic=dict(units=[64, 16]), name='ddpg-cart', noise=0.0,
             use_summary=True, optimization_steps=1, clip_norm=1.0, critic_lr=1e-3, reward_scale=1.0, seed=42)
    a.learn(300, 200, render=10)