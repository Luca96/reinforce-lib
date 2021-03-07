"""Soft Actor Critic (SAC)"""

import os
import time
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Layer, Dense, Input

from rl import utils
from rl.parameters import DynamicParameter

from rl.v2.agents import Agent
from rl.v2.memories import ReplayMemory, TransitionSpec
from rl.v2.networks import PolicyNetwork, QNetwork, backbones

from typing import Dict, Tuple, Union


# TODO: strange (time-consuming) issue during back-propagation (may be related to stop_gradient)
# TODO: try "original" implementation for policy
class SAC(Agent):
    # TODO: noise for exploration
    # TODO: `clip_reward` and/or `reward_scale` do it in `on_transition()`
    # TODO: initial_Random_steps (or batches), deterministic
    def __init__(self, *args, optimizer='adam', lr: utils.DynamicType = 3e-4, memory_size=1024, name='sac-agent',
                 actor: dict = None, critic: dict = None, clip_norm: utils.DynamicType = None, load=False,
                 polyak: utils.DynamicType = 0.995, optimization_steps=1, temperature: utils.DynamicType = 0.001,
                 target_entropy: Union[str, utils.DynamicType] = None, target_update_interval=1000, **kwargs):
        assert optimization_steps >= 1

        super().__init__(*args, name=name, **kwargs)

        # Hyper-parameters
        self.memory_size = int(memory_size)
        self.clip_norm = self._init_clip_norm(clip_norm)
        self.polyak = DynamicParameter.create(value=polyak)
        self.opt_steps = int(optimization_steps)
        self.temperature = DynamicParameter.create(value=temperature or 1.0)

        self.critic_lr = DynamicParameter.create(value=lr)
        self.actor_lr = DynamicParameter.create(value=lr)

        # Temperature adjustment
        if target_entropy is not None:
            self.should_learn_temperature = True

            if target_entropy == 'auto':
                target_entropy = 0.98 * np.log(self.num_actions)

            elif target_entropy == 'original':
                target_entropy = -self.num_actions

            self.target_entropy = DynamicParameter.create(value=target_entropy)
            self.temperature_lr = DynamicParameter.create(value=lr)
        else:
            self.should_learn_temperature = False

        # Networks
        self.weights_path = dict(actor=os.path.join(self.base_path, 'actor'),
                                 q1=os.path.join(self.base_path, 'q1'),
                                 q2=os.path.join(self.base_path, 'q2'))

        self.actor = ActorNetwork(agent=self, **(actor or {}))
        self.q1 = CriticNetwork(agent=self, **(critic or {}))
        self.q2 = CriticNetwork(agent=self, **(critic or {}))

        self.actor.compile(optimizer, clip_norm=self.clip_norm[0], learning_rate=self.actor_lr)
        self.q1.compile(optimizer, clip_norm=self.clip_norm[1], learning_rate=self.critic_lr)
        self.q2.compile(optimizer, clip_norm=self.clip_norm[1], learning_rate=self.critic_lr)

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

    # @tf.function
    def act(self, state) -> Tuple[tf.Tensor, dict, dict]:
        action, log_prob, mean, std = self.actor(state, training=True)

        other = dict()
        debug = dict(log_prob=log_prob, distribution_mean=mean, distribution_std=std)

        return action, other, debug

    def update(self):
        if not self.memory.full_enough(amount=self.batch_size):
            print('Not updated: memory is not full enough.')
            return

        t0 = time.time()
        # batches = self.memory.to_batches(**self.data_args).repeat(count=self.opt_steps)
        batches = [self.memory.sample(batch_size=self.batch_size, seed=self.seed) for _ in range(self.opt_steps)]

        for batch in batches:
            # Update Q-networks
            self.q1.train_step(batch)  # also updates `q2`

            # Update Policy
            self.actor.train_step(batch)

            # update target-nets
            self.update_target_networks()

            # update temperature
            debug = self.update_temperature(batch)
            self.log(average=True, **debug)

        print(f'Update took {round(time.time() - t0, 3)}s.')

    @tf.function
    def update_temperature(self, batch: dict) -> dict:
        if not self.should_learn_temperature:
            return {}

        alpha = tf.constant(self.temperature.value, dtype=tf.float32)
        log_prob, _ = self.actor(batch['state'], actions=batch['action'], training=False)

        with tf.GradientTape() as tape:
            tape.watch(alpha)
            loss = -alpha * (log_prob - self.target_entropy())

        grad = tape.gradient(loss, alpha)
        self.temperature.value -= self.temperature_lr() * grad

        return dict(temperature_loss=loss, temperature_gradient=grad, temperature_lr=self.temperature_lr.value,
                    target_entropy=self.target_entropy.value, temperature_log_prob=log_prob,
                    temperature=self.temperature.value)

    def update_target_networks(self):
        if self.polyak.value < 1.0:
            self.q1.update_target_network(polyak=self.polyak())
            self.q2.update_target_network(polyak=self.polyak.value)

    def learn(self, *args, **kwargs):
        t0 = time.time()
        super().learn(*args, **kwargs)
        print(f'Time {round(time.time() - t0, 3)}s.')

    def on_termination(self, last_transition, timestep: int, episode: int):
        super().on_termination(last_transition, timestep, episode)
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


class ActorNetwork(PolicyNetwork):

    @tf.function
    def objective(self, batch, reduction=tf.reduce_mean) -> tuple:
        states = batch['state']
        actions, log_prob, _, _ = self(states, training=True)

        # TODO: stop_gradient?
        # predict Q-values from both Q-functions and then take the minimum of the two
        q1 = self.agent.q1(states, actions, training=True)
        q2 = self.agent.q2(states, actions, training=True)
        # q_values = tf.expand_dims(tf.minimum(q1, q2), axis=1)
        q_values = tf.minimum(q1, q2)

        # Loss
        # entropy = tf.reduce_mean(self.agent.temperature() * log_prob)
        entropy = log_prob * self.agent.temperature()
        loss = reduction(entropy - q_values)

        return loss, dict(loss=loss, obj_entropy=entropy, q1=q1, q2=q2, q_values=q_values, obj_log_prob=log_prob,
                          temperature=self.agent.temperature.value, obj_actions=actions)


# TODO: single class that wraps two Q-networks (more elegant)
class CriticNetwork(QNetwork):
    def __init__(self, agent: SAC, log_prefix='critic', **kwargs):
        self.discrete_action_space = agent.distribution_type == 'categorical'

        super().__init__(agent, log_prefix=log_prefix, **kwargs)

    def call(self, *inputs, training=None, **kwargs):
        return super().call(inputs, actions=None, training=training)

    def act(self, inputs):
        raise NotImplementedError

    def structure(self, inputs: Dict[str, Input], name='Q-Network', **kwargs) -> tuple:
        state_in = inputs['state']
        action_in = Input(shape=(self.agent.num_actions,), name='action', dtype=tf.float32)

        x = backbones.dense_branched(state_in, action_in, **kwargs)

        output = self.output_layer(**self.output_args)(x)
        return (state_in, action_in), output, name

    @tf.function
    def objective(self, batch: dict, reduction=tf.reduce_mean) -> tuple:
        states, actions, next_states = batch['state'], batch['action'], batch['next_state']

        # Policy prediction
        next_actions, log_prob, _, _ = utils.stop_gradient(self.agent.actor(next_states))

        # Entropy
        # entropy = tf.reduce_mean(self.agent.temperature() * log_prob)
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
        # q_target = tf.expand_dims(q_target, axis=1)

        targets = tf.stop_gradient(rewards + self.agent.gamma * (1.0 - terminals) * (q_target - entropy))

        return targets, dict(target_q1=q1_target, target_q2=q2_target, target_q_min=q_target, targets=targets)

    def output_layer(self, **kwargs) -> Layer:
        if self.discrete_action_space:
            return Dense(units=self.agent.num_classes, name='q-values', **kwargs)

        return Dense(units=self.agent.num_actions, name='q-values', **kwargs)

    def train_step(self, batch: dict):
        if isinstance(batch, tuple):
            batch = batch[0]

        debug = self.train_on_batch(batch)
        self.agent.log(average=True, **({f'{self.prefix}_{k}': v for k, v in debug.items()}))

    @tf.function
    def train_on_batch(self, batch):
        with tf.GradientTape(persistent=True) as tape:
            loss_q1, loss_q2, debug = self.objective(batch)

        self.apply_gradients(tape, network=self, loss=loss_q1, debug=debug, postfix='q1')
        self.apply_gradients(tape, network=self.agent.q2, loss=loss_q2, debug=debug, postfix='q2')
        del tape

        return debug

    @staticmethod
    def apply_gradients(tape: tf.GradientTape, network: 'CriticNetwork', loss: tf.Tensor, debug: dict, postfix=''):
        trainable_vars = network.trainable_variables

        gradients = tape.gradient(loss, trainable_vars)
        debug[f'gradient_norm_{postfix}'] = [tf.norm(g) for g in gradients]

        if network.should_clip_gradients:
            gradients, global_norm = utils.clip_gradients2(gradients, norm=network.clip_norm())
            debug[f'gradient_clipped_norm_{postfix}'] = [tf.norm(g) for g in gradients]
            debug[f'gradient_global_norm_{postfix}'] = global_norm
            debug[f'clip_norm_{postfix}'] = network.clip_norm.value
            # gradients = utils.clip_gradients(gradients, norm=network.clip_norm())
            # debug[f'gradient_clipped_norm_{postfix}'] = [tf.norm(g) for g in gradients]
            # debug[f'clip_norm_{postfix}'] = network.clip_norm.value

        network.optimizer.apply_gradients(zip(gradients, trainable_vars))


class SACDiscrete:
    pass


if __name__ == '__main__':
    a = SAC(env='LunarLanderContinuous-v2', batch_size=64, actor=dict(units=128//2), critic=dict(units=[96-32, 32/2]),
            use_summary=True, temperature=0.1, target_entropy='auto', optimization_steps=2, seed=42)
    a.learn(200+300, 200)
