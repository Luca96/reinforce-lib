"""Soft Actor-Critic (SAC)"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import gym
import numpy as np
import tensorflow as tf

from tensorflow_probability import distributions as tfd
from tensorflow.keras.layers import Input, Layer, Dense, Concatenate

from rl import utils
from rl.parameters import DynamicParameter

from rl.v2.agents import Agent
from rl.v2.agents.ddpg import CriticNetwork
from rl.v2.agents.td3 import TwinCriticNetwork
from rl.layers import Linear
from rl.v2.memories import TransitionSpec, ReplayMemory
from rl.v2.networks import backbones, Network

from typing import Dict, List, Tuple


class SquashedGaussianPolicy(Network):

    def __init__(self, *args, log_std_range=(-20, 2), log_prefix='policy', **kwargs):
        self.log_std_range = (tf.constant(log_std_range[0], dtype=tf.float32),
                              tf.constant(log_std_range[1], dtype=tf.float32))
        self.init_hack()
        super().__init__(*args, log_prefix=log_prefix, **kwargs)

    @tf.function
    def call(self, inputs, deterministic=False, log_prob=False, training=None, **kwargs):
        mean, log_std = super().call(inputs, training=training, **kwargs)
        std = tf.exp(log_std)

        # define a Normal distribution
        normal = tfd.Normal(loc=mean, scale=std)

        # squashing sampled actions by tanh
        if deterministic:
            action = mean
        else:
            action = normal.sample(seed=self.agent.seed)

        # compute log-probability
        # - see: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py#L54-L60
        if log_prob:
            log_prob = tf.reduce_sum(normal.log_prob(action), axis=-1)
            log_prob -= tf.reduce_sum(2.0 * (tf.math.log(2.0) - action - tf.nn.softplus(-2.0 * action)), axis=1)

            return tf.nn.tanh(action), tf.reshape(log_prob, shape=(-1, 1))

        return tf.nn.tanh(action), mean, std  # also return `mean` and `std` for debugging

    def structure(self, inputs: Dict[str, tf.keras.Input], name='SquashedGaussianPolicy', **kwargs) -> tuple:
        return super().structure(inputs, name=name, **kwargs)

    def output_layer(self, layer: Layer, **kwargs) -> Tuple[Layer, Layer]:
        mean = Linear(units=self.agent.num_actions, name='mean', **self.output_kwargs)(layer)
        log_std = Dense(units=self.agent.num_actions, activation=self._clip_activation, name='log-std',
                        **self.output_kwargs)(layer)
        return mean, log_std

    @tf.function
    def objective(self, batch, reduction=tf.reduce_mean) -> tuple:
        states = batch['state']

        actions, log_prob = self(states, log_prob=True, training=True)
        alpha = tf.exp(self.agent.log_alpha)

        q_values1, q_values2 = self.agent.critic((states, actions), training=False)
        # q_values1 = self.agent.q1(states, actions, training=False)
        # q_values2 = self.agent.q2(states, actions, training=False)
        min_q_values = tf.minimum(q_values1, q_values2)

        loss = reduction(alpha * log_prob - min_q_values)
        return loss, dict(loss=loss, actions=actions, log_prob=log_prob, alpha=alpha)

    @tf.function
    def _clip_activation(self, x):
        return tf.minimum(tf.maximum(x, self.log_std_range[0]), self.log_std_range[1])


class SoftTwinCriticNetwork(TwinCriticNetwork):

    def call(self, *inputs, training=None, **kwargs):
        return CriticNetwork.call(self, inputs, training=training, **kwargs)

    def structure(self, inputs: Dict[str, Input], name='SAC-TwinCriticNetwork', **kwargs) -> tuple:
        return super().structure(inputs, name=name, **kwargs)

    @tf.function
    def train_on_batch(self, batch: dict):
        with tf.GradientTape(persistent=True) as tape:
            loss_q1, loss_q2, debug = self.objective(batch)

        vars_q1 = self.q1.trainable_variables
        vars_q2 = self.q2.trainable_variables

        # compute gradients
        grads_q1 = tape.gradient(loss_q1, vars_q1)
        grads_q2 = tape.gradient(loss_q2, vars_q2)
        del tape

        # debug
        debug['gradient_norm1'] = utils.tf_norm(grads_q1)
        debug['gradient_norm2'] = utils.tf_norm(grads_q2)
        debug['gradient_global_norm1'] = utils.tf_global_norm(debug['gradient_norm1'])
        debug['gradient_global_norm2'] = utils.tf_global_norm(debug['gradient_norm2'])

        debug.update({f'gradient1-{i}_hist': g for i, g in enumerate(grads_q1)})
        debug.update({f'gradient2-{i}_hist': g for i, g in enumerate(grads_q2)})

        # clip and apply gradients
        if self.should_clip_gradients:
            grads_q1 = self.clip_gradients(grads_q1, debug)
            grads_q2 = self.clip_gradients(grads_q2, debug)

        self.optimizer.apply_gradients(zip(grads_q1, vars_q1))
        self.optimizer.apply_gradients(zip(grads_q2, vars_q2))

        return (loss_q1 + loss_q2) / 2.0, debug

    @tf.function
    def objective(self, batch: dict, reduction=tf.reduce_mean) -> tuple:
        actions = batch['action']
        states = batch['state']

        q_values1, q_values2 = self((states, actions), training=True)
        q_targets, debug = self.targets(batch)

        # compute losses
        q1_loss = reduction(0.5 * tf.square(q_values1 - q_targets))
        q2_loss = reduction(0.5 * tf.square(q_values2 - q_targets))

        debug.update(loss_q1=q1_loss, loss_q2=q2_loss, q_values1=q_values1, q_values2=q_values2)
        return q1_loss, q2_loss, debug

    @tf.function
    def targets(self, batch: dict):
        next_states = batch['next_state']

        next_actions, log_prob = self.agent.actor(next_states, log_prob=True, training=False)
        alpha = tf.exp(self.agent.log_alpha)

        q_values1, q_values2 = self.target((next_states, next_actions), training=False)
        next_q_values = tf.minimum(q_values1, q_values2) - alpha * log_prob

        targets = batch['reward'] + self.agent.gamma * next_q_values * (1.0 - batch['terminal'])
        targets = tf.stop_gradient(targets)

        return targets, dict(targets=targets, next_q_values=next_q_values)


# TODO: support PER? or generic memory support?
# TODO: inherit from DDPG?
class SAC(Agent):

    def __init__(self, *args, name='sac', entropy_lr=1e-3, target_entropy: utils.DynamicType = None, actor_lr=1e-4,
                 actor: dict = None, critic_lr=1e-4, critic: dict = None, memory_size=1024, polyak=0.995,
                 entropy=1e-3, clip_norm=(None, None), optimizer='adam', **kwargs):
        assert memory_size >= 1
        assert 0 < polyak <= 1

        super().__init__(*args, name=name, **kwargs)

        # hyper-parameters
        self.memory_size = int(memory_size)
        self.polyak = float(polyak)
        self.entropy_weight = float(entropy)

        self.entropy_lr = DynamicParameter.create(value=entropy_lr)
        self.critic_lr = DynamicParameter.create(value=critic_lr)
        self.actor_lr = DynamicParameter.create(value=actor_lr)

        self.log_alpha = tf.Variable(initial_value=0.0, trainable=True, dtype=tf.float32)
        self.alpha_optimizer = utils.get_optimizer(optimizer, learning_rate=self.entropy_lr)

        if target_entropy is None:
            self.target_entropy = DynamicParameter.create(value=-np.prod(self.action_high.shape))

        # Networks
        self.weights_path = dict(actor=os.path.join(self.base_path, 'actor'),
                                 # q1=os.path.join(self.base_path, 'q1'),
                                 # q2=os.path.join(self.base_path, 'q2'),
                                 critic=os.path.join(self.base_path, 'critic'))

        self.actor = Network.create(agent=self, **(actor or {}), target=False, base_class=SquashedGaussianPolicy)
        # self.q1 = Network.create(agent=self, **(critic or {}), target=True, base_class=CriticNetwork)
        # self.q2 = Network.create(agent=self, **(critic or {}), target=True, base_class=CriticNetwork)
        self.critic = Network.create(agent=self, **(critic or {}), target=True, base_class=SoftTwinCriticNetwork)

        self.actor.compile(optimizer, clip_norm=clip_norm[0], clip=self.clip_grads, learning_rate=self.actor_lr)
        # self.q1.compile(optimizer, clip_norm=clip_norm[1], clip=self.clip_grads, learning_rate=self.critic_lr)
        # self.q2.compile(optimizer, clip_norm=clip_norm[1], clip=self.clip_grads, learning_rate=self.critic_lr)
        self.critic.compile(optimizer, clip_norm=clip_norm[1], clip=self.clip_grads, learning_rate=self.critic_lr)

    @property
    def transition_spec(self) -> TransitionSpec:
        return TransitionSpec(state=self.state_spec, action=(self.num_actions,), next_state=True, terminal=True)

    def define_memory(self) -> ReplayMemory:
        return ReplayMemory(self.transition_spec, shape=self.memory_size, seed=self.seed)

    def _init_action_space(self):
        action_space = self.env.action_space

        assert isinstance(action_space, gym.spaces.Box)
        assert action_space.is_bounded()

        self.action_low = tf.constant(action_space.low, dtype=tf.float32)
        self.action_high = tf.constant(action_space.high, dtype=tf.float32)
        self.action_range = tf.constant(action_space.high - action_space.low, dtype=tf.float32)

        self.num_actions = action_space.shape[0]

        # `a` \in (-1, 1), so add 1 and divide by 2 (to rescale in 0-1)
        self.convert_action = lambda a: tf.squeeze((a + 1.0) / 2.0 * self.action_range + self.action_low).numpy()

    @tf.function
    def act(self, state, deterministic=False, **kwargs) -> Tuple[tf.Tensor, dict, dict]:
        action, mean, std = self.actor(state, deterministic=deterministic, **kwargs)
        debug = dict(distribution_mean=mean, distribution_std=std)

        return action, {}, debug

    @tf.function
    def act_randomly(self, state) -> Tuple[tf.Tensor, dict, dict]:
        action, _, _ = self.actor(state, deterministic=False)
        return action, {}, {}

    def learn(self, *args, **kwargs):
        with utils.Timed('Learn'):
            super().learn(*args, **kwargs)

    def update(self):
        batch = self.memory.get_batch(batch_size=self.batch_size)

        self.update_alpha(batch)
        self.critic.train_step(batch)
        self.actor.train_step(batch)

        self.critic.update_target_network(polyak=self.polyak)

    @tf.function
    def update_alpha(self, batch):
        with tf.GradientTape() as tape:
            _, log_prob = self.actor(batch['state'], log_prob=True, training=False)

            target_alpha = tf.stop_gradient(log_prob + self.target_entropy())
            alpha_loss = -tf.reduce_mean(self.log_alpha * target_alpha)

        grads = tape.gradient(alpha_loss, self.log_alpha)
        self.alpha_optimizer.apply_gradients(zip([grads], [self.log_alpha]))

        # debug
        self.log(alpha_loss=alpha_loss, alpha_target=target_alpha, alpha_gradient=grads)

    def on_transition(self, *args, exploration=False):
        super().on_transition(*args, exploration=exploration)

        if not exploration:
            self.update()

    # def save_weights(self):
    #     self.actor.save_weights(filepath=self.weights_path['actor'])
    #     # self.q1.save_weights(filepath=self.weights_path['q1'])
    #     # self.q2.save_weights(filepath=self.weights_path['q2'])
    #     self.critic.save_weights(filepath=self.weights_path['critic'])
    #
    # def load_weights(self):
    #     self.actor.load_weights(filepath=self.weights_path['actor'], by_name=False)
    #     # self.q1.load_weights(filepath=self.weights_path['q1'], by_name=False)
    #     # self.q2.load_weights(filepath=self.weights_path['q2'], by_name=False)
    #     self.critic.load_weights(filepath=self.weights_path['critic'], by_name=False)
    #
    # def summary(self):
    #     self.actor.summary()
    #     # self.q1.summary()
    #     self.critic.summary()


if __name__ == '__main__':
    utils.set_random_seed(42)

    agent = SAC(env='Pendulum-v1', actor_lr=3e-4, critic_lr=5e-4, polyak=0.995,
                memory_size=100_000, batch_size=256, name='sac-pendulum', use_summary=True,
                actor=dict(units=256), critic=dict(units=256), seed=utils.GLOBAL_SEED)

    # agent.summary()
    # breakpoint()

    # fix specific to pendulum env
    agent._convert_action = agent.convert_action
    agent.convert_action = lambda a: np.reshape(agent._convert_action(a), newshape=[1])

    agent.learn(episodes=200, timesteps=200, evaluation=dict(freq=10, episodes=20),
                exploration_steps=5 * agent.batch_size, save=True)
    exit()

    import pybullet_envs

    agent = SAC(env='HalfCheetahBulletEnv-v0', actor_lr=3e-4, critic_lr=5e-4, polyak=0.999,
                memory_size=100_000, batch_size=256, name='sac-cheetah', use_summary=False,
                actor=dict(units=256 // 2), critic=dict(units=256 // 2))

    agent.learn(episodes=1000, timesteps=200, evaluation=dict(freq=100, episodes=50, timesteps=1000),
                save=False, exploration_steps=5 * agent.batch_size)
