"""Deep Deterministic Policy Gradient (DDPG) Agent"""

import os
import tensorflow as tf

from tensorflow.keras.layers import Layer, Dense, Input

from rl import utils
from rl.parameters import DynamicParameter

from rl.v2.agents import Agent
from rl.v2.memories import PrioritizedMemory, ReplayMemory, TransitionSpec
from rl.v2.networks import Network, DeterministicPolicyNetwork, QNetwork, backbones

from typing import Dict, Tuple, Union


class DDPG(Agent):
    def __init__(self, *args, optimizer='adam', actor_lr: utils.DynamicType = 1e-4, critic_lr: utils.DynamicType = 1e-3,
                 memory_size=1024, name='ddpg-agent', actor: dict = None, critic: dict = None, load=False,
                 clip_norm: utils.DynamicType = None, polyak: utils.DynamicType = 0.999, optimization_steps=1,
                 noise: utils.DynamicType = 0.05, prioritized=False, alpha: utils.DynamicType = 0.6,
                 beta: utils.DynamicType = 0.4, **kwargs):
        assert optimization_steps >= 1

        super().__init__(*args, name=name, **kwargs)

        # Hyper-parameters
        self.memory_size = int(memory_size)
        self.clip_norm = self._init_clip_norm(clip_norm)
        self.polyak = DynamicParameter.create(value=polyak)
        self.optimization_steps = int(optimization_steps)
        self.noise = DynamicParameter.create(value=noise)
        self.prioritized = bool(prioritized)

        self.critic_lr = DynamicParameter.create(value=critic_lr)
        self.actor_lr = DynamicParameter.create(value=actor_lr)

        # PER memory params:
        if self.prioritized:
            self.alpha = DynamicParameter.create(value=alpha)
            self.beta = DynamicParameter.create(value=beta)

        # Networks
        self.weights_path = dict(actor=os.path.join(self.base_path, 'actor'),
                                 critic=os.path.join(self.base_path, 'critic'))

        self.actor = Network.create(agent=self, target=True, log_prefix='actor', **(actor or {}),
                                    base_class='DDPG-ActorNetwork')
        self.critic = Network.create(agent=self, log_prefix='critic', **(critic or {}), base_class='DDPG-CriticNetwork')

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
    def memory(self) -> Union[ReplayMemory, PrioritizedMemory]:
        if self._memory is None:
            if self.prioritized:
                self._memory = PrioritizedMemory(self.transition_spec, shape=self.memory_size, alpha=self.alpha,
                                                 beta=self.beta, seed=self.seed)
            else:
                self._memory = ReplayMemory(self.transition_spec, shape=self.memory_size, seed=self.seed)

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
                action = tf.clip_by_value(action, -1.0, 1.0)
                action = (action + 1.0) / 2.0  # transform to [0, 1] (as Beta would do)
                return tf.squeeze(action * action_range + action_low).numpy()

            self.convert_action = convert_action

    # @tf.function
    def act(self, state) -> Tuple[tf.Tensor, dict, dict]:
        action = self.actor(state, training=True)

        if self.noise.value > 0.0:
            noise = tf.random.normal(shape=action.shape, stddev=self.noise(), seed=self.seed)
            debug = dict(noise=noise, noise_std=self.noise.value, action_without_noise=action)
        else:
            debug = {}
            noise = 0.0

        # TODO: should clip actions + noise?
        return action + noise, {}, debug

    def act_randomly(self, state) -> Tuple[tf.Tensor, dict, dict]:
        if self.distribution_type == 'categorical':
            # sample `logits` instead of actions
            action = tf.random.uniform(shape=(self.num_classes,), seed=self.seed)

        elif self.distribution_type == 'beta':
            action = tf.random.uniform(shape=(self.num_actions,), minval=-1.0, maxval=1.0, seed=self.seed)
        else:
            action = tf.random.normal(shape=(self.num_actions,), seed=self.seed)

        return action, {}, {}

    def update(self):
        if not self.memory.full_enough(amount=self.batch_size):
            return self.memory.update_warning(self.batch_size)

        for _ in range(self.optimization_steps):
            batch = self.memory.sample(batch_size=self.batch_size)

            self.critic.train_step(batch)
            self.actor.train_step(batch)

            self.update_target_networks()

            if self.prioritized:
                self.memory.update_priorities()

    def update_target_networks(self):
        if self.polyak.value < 1.0:
            self.actor.update_target_network(polyak=self.polyak())
            self.critic.update_target_network(polyak=self.polyak.value)

    def learn(self, *args, **kwargs):
        with utils.Timed('Learn'):
            super().learn(*args, **kwargs)

    def on_transition(self, transition: dict, timestep: int, episode: int, exploration=False):
        super().on_transition(transition, timestep, episode, exploration)

        if not exploration:
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


@Network.register(name='DDPG-ActorNetwork')
class ActorNetwork(DeterministicPolicyNetwork):

    @tf.function
    def objective(self, batch: dict, reduction=tf.reduce_mean) -> tuple:
        states = batch['state']
        actions = self(states, training=True)

        q_values = self.agent.critic(states, actions, training=False)
        loss = -reduction(q_values)

        return loss, dict(loss=loss, actions=actions, q_values=q_values)


@Network.register(name='DDPG-CriticNetwork')
class CriticNetwork(QNetwork):

    def call(self, *inputs, training=None, **kwargs):
        return super().call(inputs, actions=None, training=training)

    def act(self, inputs):
        raise NotImplementedError

    def structure(self, inputs: Dict[str, Input], name='CriticNetwork', **kwargs) -> tuple:
        utils.remove_keys(kwargs, ['dueling', 'operator', 'prioritized'])
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

    actor = dict(units=32, output=dict(kernel_initializer='glorot_uniform', bias_initializer='zeros'))

    a = DDPG(env=env2, batch_size=64, name='ddpg-cart', noise=0.01, use_summary=True,
             actor=actor, critic=dict(units=[64, 16]), memory_size=4096, polyak=0.995, prioritized=True,
             optimization_steps=1, clip_norm=1.0, critic_lr=1e-3, reward_scale=1.0 / 4, seed=42)
    a.learn(250, 200, exploration_steps=4096)
