"""Soft Actor Critic (SAC)"""

import os
import time
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Layer, Dense, Input

from rl import utils
from rl.parameters import DynamicParameter

from rl.v2.agents import Agent
from rl.v2.memories import PrioritizedMemory, ReplayMemory, TransitionSpec
from rl.v2.networks import Network, PolicyNetwork, QNetwork, backbones

from typing import Dict, Tuple, Union


# TODO: try "original" implementation for policy
class SAC(Agent):
    # TODO: noise for exploration
    # TODO: initial_Random_steps (or batches), deterministic
    def __init__(self, *args, optimizer='adam', lr: utils.DynamicType = 3e-4, memory_size=1024, name='sac-agent',
                 actor: dict = None, critic: dict = None, clip_norm: utils.DynamicType = None, load=False,
                 polyak: utils.DynamicType = 0.995, optimization_steps=1, temperature: utils.DynamicType = 0.001,
                 target_entropy: Union[str, utils.DynamicType] = None, target_update_interval=1000, prioritized=False,
                 alpha: utils.DynamicType = 0.6, beta: utils.DynamicType = 0.4, **kwargs):
        assert optimization_steps >= 1

        super().__init__(*args, name=name, **kwargs)
        self.horizon = 1  # for compatibility with Q-network and N-step estimator

        # Hyper-parameters
        self.memory_size = int(memory_size)
        self.clip_norm = self._init_clip_norm(clip_norm)
        self.polyak = DynamicParameter.create(value=polyak)
        self.opt_steps = int(optimization_steps)
        self.temperature = DynamicParameter.create(value=temperature or 1.0)
        self.prioritized = bool(prioritized)

        self.critic_lr = DynamicParameter.create(value=lr)
        self.actor_lr = DynamicParameter.create(value=lr)

        # PER memory params:
        if self.prioritized:
            self.alpha = DynamicParameter.create(value=alpha)
            self.beta = DynamicParameter.create(value=beta)

        # Temperature adjustment
        if target_entropy is not None:
            self.should_learn_temperature = True

            if target_entropy == 'auto':
                target_entropy = 0.98 * np.log(self.num_actions)

            elif target_entropy == 'original':
                # target_entropy = -self.num_actions
                target_entropy = self.num_actions

            self.target_entropy = DynamicParameter.create(value=target_entropy)
            self.temperature_lr = DynamicParameter.create(value=lr)
        else:
            self.should_learn_temperature = False

        # Networks
        self.weights_path = dict(actor=os.path.join(self.base_path, 'actor'),
                                 q1=os.path.join(self.base_path, 'q1'),
                                 q2=os.path.join(self.base_path, 'q2'))

        self.actor = Network.create(agent=self, **(actor or {}), base_class='SAC-ActorNetwork')
        self.q1 = Network.create(agent=self, **(critic or {}), base_class='SAC-CriticNetwork')
        self.q2 = Network.create(agent=self, **(critic or {}), base_class='SAC-CriticNetwork')

        self.actor.compile(optimizer, clip_norm=self.clip_norm[0], clip=self.clip_grads, learning_rate=self.actor_lr)
        self.q1.compile(optimizer, clip_norm=self.clip_norm[1], clip=self.clip_grads, learning_rate=self.critic_lr)
        self.q2.compile(optimizer, clip_norm=self.clip_norm[1], clip=self.clip_grads, learning_rate=self.critic_lr)

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

    def _init_clip_norm(self, clip_norm):
        if clip_norm is None:
            return None, None

        if isinstance(clip_norm, float) or isinstance(clip_norm, int):
            return clip_norm, clip_norm

        if isinstance(clip_norm, tuple):
            assert 0 < len(clip_norm) <= 2

            if len(clip_norm) < 2:
                return clip_norm[0], clip_norm[0]
            else:
                return clip_norm

        raise ValueError(f'Parameter "clip_norm" should be `int`, `float` or `tuple` not {type(clip_norm)}.')

    @tf.function
    def act(self, state) -> Tuple[tf.Tensor, dict, dict]:
        action, log_prob, mean, std = self.actor(state, training=True)

        other = dict()
        debug = dict(log_prob=log_prob, distribution_mean=mean, distribution_std=std)

        return action, other, debug

    def update(self):
        if not self.memory.full_enough(amount=self.batch_size):
            return self.memory.update_warning(self.batch_size)

        with utils.Timed('Update'):
            for _ in range(self.opt_steps):
                batch = self.memory.sample(batch_size=self.batch_size)

                # update Q-networks
                self.q1.train_step(batch)  # also updates `q2`

                # update Policy
                self.actor.train_step(batch)

                # update target-networks
                self.update_target_networks()

                # update temperature
                debug = self.update_temperature(batch)
                self.log(average=True, **debug)

                if self.prioritized:
                    self.memory.update_priorities()

    @tf.function
    def update_temperature(self, batch: dict) -> dict:
        if not self.should_learn_temperature:
            return {}

        # alpha = tf.constant(self.temperature.value, dtype=tf.float32)
        log_prob, _ = self.actor(batch['state'], actions=batch['action'], training=False)

        with tf.GradientTape() as tape:
            # tape.watch(alpha)
            tape.watch(self.temperature.variable)
            loss = -self.temperature.value * tf.stop_gradient(log_prob - self.target_entropy())

        grad = tape.gradient(loss, self.temperature.variable)
        self.temperature.value -= self.temperature_lr() * grad

        return dict(temperature_loss=loss, temperature_gradient=grad, temperature_lr=self.temperature_lr.value,
                    target_entropy=self.target_entropy.value, temperature_log_prob=log_prob,
                    temperature=self.temperature.value)

    def update_target_networks(self):
        if self.polyak.value < 1.0:
            self.q1.update_target_network(polyak=self.polyak())
            self.q2.update_target_network(polyak=self.polyak.value)

    def learn(self, *args, **kwargs):
        with utils.Timed('Learn'):
            super().learn(*args, **kwargs)

    def on_termination(self, last_transition, timestep: int, episode: int, exploration=False):
        super().on_termination(last_transition, timestep, episode, exploration)

        if not exploration:
            self.update()

    def save_weights(self):
        self.actor.save_weights(filepath=self.weights_path['actor'])
        self.q1.save_weights(filepath=self.weights_path['q1'])
        self.q2.save_weights(filepath=self.weights_path['q2'])

    def load_weights(self):
        self.actor.load_weights(filepath=self.weights_path['actor'], by_name=False)
        self.q1.load_weights(filepath=self.weights_path['q1'], by_name=False)
        self.q2.load_weights(filepath=self.weights_path['q2'], by_name=False)

    def summary(self):
        self.actor.summary()
        self.q1.summary()
        # q2 is same as q1


@Network.register(name='SAC-ActorNetwork')
class ActorNetwork(PolicyNetwork):

    @tf.function
    def objective(self, batch, reduction=tf.reduce_mean) -> tuple:
        states = batch['state']
        actions, log_prob, _, _ = self(states, training=True)

        # predict Q-values from both Q-functions and then take the minimum of the two
        q1 = self.agent.q1(states, actions, training=True)
        q2 = self.agent.q2(states, actions, training=True)
        q_values = tf.minimum(q1, q2)

        # Loss
        entropy = log_prob * self.agent.temperature.value
        loss = reduction(entropy - q_values)

        return loss, dict(loss=loss, obj_entropy=entropy, q1=q1, q2=q2, q_values=q_values, obj_log_prob=log_prob,
                          temperature=self.agent.temperature.value, obj_actions=actions)


@Network.register(name='SAC-CriticNetwork')
class CriticNetwork(QNetwork):
    def __init__(self, agent: SAC, log_prefix='critic', **kwargs):
        self.discrete_action_space = agent.distribution_type == 'categorical'

        super().__init__(agent, log_prefix=log_prefix, **kwargs)

    def call(self, *inputs, training=None, **kwargs):
        return super().call(inputs, actions=None, training=training)

    def act(self, inputs):
        raise NotImplementedError

    def structure(self, inputs: Dict[str, Input], name='SAC-Q-Network', **kwargs) -> tuple:
        utils.remove_keys(kwargs, keys=['dueling', 'operator', 'prioritized'])

        state_in = inputs['state']
        action_in = Input(shape=(self.agent.num_actions,), name='action', dtype=tf.float32)

        x = backbones.dense_branched(state_in, action_in, **kwargs)

        output = self.output_layer(layer=x)
        # output = self.output_layer(**self.output_args)(x)
        return (state_in, action_in), output, name

    @tf.function
    def objective(self, batch: dict, reduction=tf.reduce_mean) -> tuple:
        states, actions, next_states = batch['state'], batch['action'], batch['next_state']

        # Policy prediction
        next_actions, log_prob, _, _ = utils.stop_gradient(self.agent.actor(next_states))

        # Entropy
        entropy = tf.stop_gradient(log_prob * self.agent.temperature())

        # Targets
        targets, debug = self.targets(states=next_states, actions=next_actions, rewards=batch['reward'],
                                      terminals=batch['terminal'], entropy=entropy)
        # Q-loss
        q1 = self(states, actions, training=True)
        q2 = self.agent.q2(states, actions, training=True)

        loss_q1 = 0.5 * reduction(tf.square(q1 - targets))
        loss_q2 = 0.5 * reduction(tf.square(q2 - targets))

        # debug
        debug.update(loss_q1=loss_q1, loss_q2=loss_q2, q_values1=q1, q_values2=q2, entropy=entropy,
                     temperature=self.agent.temperature.value)

        return loss_q1, loss_q2, debug

    @tf.function
    def targets(self, states, actions, rewards, terminals, entropy):
        q1_target = self.target(states, actions, training=False)
        q2_target = self.agent.q2.target(states, actions, training=False)

        q_target = tf.minimum(q1_target, q2_target)
        targets = tf.stop_gradient(rewards + self.agent.gamma * (1.0 - terminals) * (q_target - entropy))

        return targets, dict(target_q1=q1_target, target_q2=q2_target, target_q_min=q_target, targets=targets)

    def output_layer(self, layer) -> Layer:
        if self.discrete_action_space:
            return super().output_layer(layer)  # compatible with "dueling" architecture
            # return Dense(units=self.agent.num_classes, name='q-values', **kwargs)

        # TODO: think about a "continuous" dueling architecture; see paper
        return Dense(units=self.agent.num_actions, name='q-values', **self.output_args)(layer)

    def train_step(self, batch: dict):
        if isinstance(batch, tuple):
            batch = batch[0]

        debug = self.train_on_batch(batch)
        self.agent.log(average=True, **({f'{self.prefix}_{k}': v for k, v in debug.items()}))

    @tf.function
    def train_on_batch(self, batch):
        with tf.GradientTape() as tape:
            loss_q1, loss_q2, debug = self.objective(batch)
            loss = loss_q1 + loss_q2

        v = self.trainable_variables + self.agent.q2.trainable_variables
        gradients = tape.gradient(loss, v)

        debug['gradient_norm'] = utils.tf_norm(gradients)
        debug['gradient_global_norm'] = utils.tf_global_norm(debug['gradient_norm'])

        if self.should_clip_gradients:
            gradients, _ = utils.clip_gradients_global(gradients, norm=self.clip_norm())
            debug['gradient_clipped_norm'] = utils.tf_norm(gradients)
            debug['clip_norm'] = self.clip_norm.value

        self.optimizer.apply_gradients(zip(gradients, v))
        return debug


class SACDiscrete:
    pass

