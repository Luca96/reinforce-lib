"""Soft Actor-Critic (SAC)
    - Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor (arXiv:1801.01290)
    - Soft Actor-Critic Algorithms and Applications (arXiv:1812.05905)
"""
import gym.spaces
import numpy as np
import tensorflow as tf

from tensorflow_probability import distributions as tfd
from tensorflow.keras.layers import Input, Layer, Dense, Concatenate

from rl import utils
from rl.parameters import DynamicParameter

from rl.agents import Agent
from rl.agents.ddpg import CriticNetwork
from rl.agents.td3 import TwinCriticNetwork
from rl.agents.actions import TanhConverter, DiscreteConverter
from rl.layers import Linear, distributions as rld
from rl.memories import TransitionSpec, ReplayMemory, PrioritizedMemory
from rl.networks import Network

from typing import Dict, Union, Tuple


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

    def output_layer(self, layer: Layer, **kwargs) -> Tuple[Layer, Layer]:
        mean = Linear(units=self.agent.num_actions, name='mean', **kwargs)(layer)
        log_std = Dense(units=self.agent.num_actions, activation=self._clip_activation, name='log-std',
                        **kwargs)(layer)
        return mean, log_std

    @tf.function
    def objective(self, batch, reduction=tf.reduce_mean) -> tuple:
        states = batch['state']

        actions, log_prob = self(states, log_prob=True, training=True)
        alpha = tf.exp(self.agent.log_alpha)

        q_values1, q_values2 = self.agent.critic((states, actions), training=False)
        min_q_values = tf.minimum(q_values1, q_values2)

        loss = reduction(alpha * log_prob - min_q_values)
        return loss, dict(loss=loss, actions=actions, log_prob=log_prob, alpha=alpha)

    @tf.function
    def _clip_activation(self, x):
        return tf.minimum(tf.maximum(x, self.log_std_range[0]), self.log_std_range[1])


class SoftTwinCriticNetwork(TwinCriticNetwork):

    def call(self, *inputs, training=None, **kwargs):
        return CriticNetwork.call(self, inputs, training=training, **kwargs)

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


class SAC(Agent):

    def __init__(self, *args, name='sac', entropy_lr=1e-3, target_entropy: utils.DynamicType = None, actor_lr=1e-4,
                 actor: dict = None, critic_lr=1e-4, critic: dict = None, memory_size=1024, optimizer='adam',
                 entropy=1e-3, prioritized=False, alpha: utils.DynamicType = 0.6, beta: utils.DynamicType = 0.1,
                 clip_norm=(None, None), polyak: utils.DynamicType = 0.995, **kwargs):
        assert memory_size >= 1
        assert 0 < polyak <= 1

        super().__init__(*args, name=name, **kwargs)
        self.num_actions = self.action_converter.num_actions

        # hyper-parameters
        self.memory_size = int(memory_size)
        self.prioritized = bool(prioritized)
        self.polyak = DynamicParameter.create(value=polyak)
        self.entropy_weight = float(entropy)

        self.entropy_lr = DynamicParameter.create(value=entropy_lr)
        self.critic_lr = DynamicParameter.create(value=critic_lr)
        self.actor_lr = DynamicParameter.create(value=actor_lr)

        self.log_alpha = tf.Variable(initial_value=0.0, trainable=True, dtype=tf.float32)
        self.alpha_optimizer = utils.get_optimizer(optimizer, learning_rate=self.entropy_lr)

        if target_entropy is None:
            self.target_entropy = DynamicParameter.create(value=-np.prod(self.action_converter.action_high.shape))
        else:
            self.target_entropy = DynamicParameter.create(value=float(target_entropy))

        # PER memory params:
        if self.prioritized:
            self.alpha = DynamicParameter.create(value=alpha)
            self.beta = DynamicParameter.create(value=beta)

        # Networks
        self.actor = Network.create(agent=self, **(actor or {}), target=False, base_class=SquashedGaussianPolicy)
        self.critic = Network.create(agent=self, **(critic or {}), target=True, base_class=SoftTwinCriticNetwork)

        self.actor.compile(optimizer, clip_norm=clip_norm[0], clip=self.clip_grads, learning_rate=self.actor_lr)
        self.critic.compile(optimizer, clip_norm=clip_norm[1], clip=self.clip_grads, learning_rate=self.critic_lr)

    @property
    def transition_spec(self) -> TransitionSpec:
        return TransitionSpec(state=self.state_spec, action=self.num_actions, next_state=True, terminal=True)

    def define_memory(self) -> Union[ReplayMemory, PrioritizedMemory]:
        if self.prioritized:
            return PrioritizedMemory(self.transition_spec, shape=self.memory_size, gamma=self.gamma,
                                     alpha=self.alpha, beta=self.beta, seed=self.seed)

        return ReplayMemory(self.transition_spec, shape=self.memory_size, seed=self.seed)

    def define_action_converter(self, kwargs: dict) -> TanhConverter:
        return TanhConverter(space=self.env.action_space, **(kwargs or {}))

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
        if not self.memory.full_enough(amount=self.batch_size):
            return self.memory.update_warning(self.batch_size)

        batch = self.memory.get_batch(batch_size=self.batch_size)

        self.update_alpha(batch)
        self.critic.train_step(batch)
        self.actor.train_step(batch)

        self.update_target_networks()

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

    def update_target_networks(self):
        self.critic.update_target_network(polyak=self.polyak())
        self.log(target_critic_distance=self.critic.debug_target_network())

    def on_transition(self, *args, exploration=False):
        super().on_transition(*args, exploration=exploration)

        if not exploration:
            self.update()


# -------------------------------------------------------------------------------------------------
# -- SAC-Discrete
# -------------------------------------------------------------------------------------------------

class DiscreteTwinCriticNetwork(SoftTwinCriticNetwork):

    @tf.function
    def objective(self, batch: dict, reduction=tf.reduce_mean) -> tuple:
        actions = batch['action']
        states = batch['state']

        q_values1, q_values2 = self(states, training=True)
        q_values1 = utils.index_tensor(q_values1, indices=actions)
        q_values2 = utils.index_tensor(q_values2, indices=actions)

        q_targets, debug = self.targets(batch)
        q_targets = tf.squeeze(q_targets)

        # compute losses
        q1_loss = reduction(0.5 * tf.square(q_values1 - q_targets))
        q2_loss = reduction(0.5 * tf.square(q_values2 - q_targets))

        debug.update(loss_q1=q1_loss, loss_q2=q2_loss, q_values1=q_values1, q_values2=q_values2)
        return q1_loss, q2_loss, debug

    @tf.function
    def targets(self, batch: dict):
        next_states = batch['next_state']

        log_prob, probs = self.agent.actor(next_states, prob=True, training=False)
        alpha = tf.exp(self.agent.log_alpha)

        # https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/actor_critic_agents/SAC_Discrete.py#L71-L73
        q_values1, q_values2 = self.target(next_states, training=False)
        next_q_values = tf.minimum(q_values1, q_values2) - alpha * log_prob
        next_q_values = tf.reduce_sum(probs * next_q_values, axis=1, keepdims=True)

        targets = batch['reward'] + self.agent.gamma * next_q_values * (1.0 - batch['terminal'])
        targets = tf.stop_gradient(targets)

        return targets, dict(targets=targets, next_q_values=next_q_values)

    def structure(self, inputs: Dict[str, Input], **kwargs) -> tuple:
        self.output_kwargs['name'] = 'q1-values'
        _, output1 = Network.structure(self, inputs, **kwargs)

        self.output_kwargs['name'] = 'q2-values'
        inputs, output2 = Network.structure(self, inputs, **kwargs)

        # create two networks
        self.q1 = tf.keras.Model(inputs, outputs=output1, name=super().default_name + '-Q1')
        self.q2 = tf.keras.Model(inputs, outputs=output2, name=super().default_name + '-Q2')

        # return joint model
        return inputs, (self.q1.output, self.q2.output)

    def output_layer(self, layer: Layer, **kwargs) -> Layer:
        # now, output |A| Q-values
        return Dense(units=self.agent.action_converter.num_classes, **kwargs)(layer)

    def get_inputs(self) -> Dict[str, Input]:
        # remove the "action" Input
        return Network.get_inputs(self)


class SoftmaxPolicyNetwork(Network):

    @tf.function
    def call(self, inputs, prob=False, **kwargs):
        cat: rld.Categorical = super().call(inputs, **kwargs)
        actions = tf.identity(cat)

        if prob:
            # return probs for all (discrete) actions, as well as sampled actions
            probs = tf.reshape(cat.probs_parameter(), shape=(-1, self.agent.action_converter.num_classes))
            return cat.log_prob(actions), probs

        return actions, cat.mean(), cat.stddev()

    @tf.function
    def objective(self, batch, reduction=tf.reduce_mean) -> tuple:
        states = batch['state']

        log_prob, probs = self(states, prob=True, training=True)
        alpha = tf.exp(self.agent.log_alpha)

        q_values1, q_values2 = self.agent.critic(states, training=False)
        min_q_values = tf.minimum(q_values1, q_values2)

        # https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/actor_critic_agents/SAC_Discrete.py#L81
        inside_term = alpha * log_prob - min_q_values
        loss = reduction(tf.reduce_sum(probs * inside_term, axis=1))

        return loss, dict(loss=loss, log_prob=log_prob, alpha=alpha)

    def output_layer(self, layer: Layer, **kwargs) -> rld.DistributionLayer:
        action_space = self.agent.env.action_space
        assert isinstance(action_space, gym.spaces.Discrete)

        return rld.Categorical(num_actions=1, num_classes=action_space.n, **kwargs)(layer)


class SACDiscrete(SAC):
    """SAC-Discrete:
        - Soft Actor-Critic for Discrete Action Settings (arxiv.org/abs/1910.07207)
    """

    def __init__(self, *args, name='sac_discrete', entropy_lr=1e-3, target_entropy: utils.DynamicType = None, actor_lr=1e-4,
                 actor: dict = None, critic_lr=1e-4, critic: dict = None, memory_size=1024, optimizer='adam',
                 entropy=1e-3, prioritized=False, alpha: utils.DynamicType = 0.6, beta: utils.DynamicType = 0.1,
                 clip_norm=(None, None), polyak: utils.DynamicType = 0.995, **kwargs):
        assert memory_size >= 1
        assert 0 < polyak <= 1

        Agent.__init__(self, *args, name=name, **kwargs)
        self.num_actions = self.action_converter.num_actions

        # hyper-parameters
        self.memory_size = int(memory_size)
        self.prioritized = bool(prioritized)
        self.polyak = DynamicParameter.create(value=polyak)
        self.entropy_weight = float(entropy)

        self.entropy_lr = DynamicParameter.create(value=entropy_lr)
        self.critic_lr = DynamicParameter.create(value=critic_lr)
        self.actor_lr = DynamicParameter.create(value=actor_lr)

        self.log_alpha = tf.Variable(initial_value=0.0, trainable=True, dtype=tf.float32)
        self.alpha_optimizer = utils.get_optimizer(optimizer, learning_rate=self.entropy_lr)

        if target_entropy is None:
            target_value = -np.log((1.0 / self.action_converter.num_classes)) * 0.98
            self.target_entropy = DynamicParameter.create(value=target_value)
        else:
            self.target_entropy = DynamicParameter.create(value=float(target_entropy))

        # PER memory params:
        if self.prioritized:
            self.alpha = DynamicParameter.create(value=alpha)
            self.beta = DynamicParameter.create(value=beta)

        # Networks
        self.actor = Network.create(agent=self, **(actor or {}), target=False, base_class=SoftmaxPolicyNetwork)
        self.critic = Network.create(agent=self, **(critic or {}), target=True, base_class=DiscreteTwinCriticNetwork)

        self.actor.compile(optimizer, clip_norm=clip_norm[0], clip=self.clip_grads, learning_rate=self.actor_lr)
        self.critic.compile(optimizer, clip_norm=clip_norm[1], clip=self.clip_grads, learning_rate=self.critic_lr)

    def define_action_converter(self, kwargs: dict) -> DiscreteConverter:
        return DiscreteConverter(space=self.env.action_space, **(kwargs or {}))

    @tf.function
    def update_alpha(self, batch):
        with tf.GradientTape() as tape:
            log_prob, probs = self.actor(batch['state'], prob=True, training=False)

            target_alpha = tf.stop_gradient(log_prob + self.target_entropy())
            alpha_loss = tf.reduce_sum(probs * self.log_alpha * target_alpha, axis=1)
            alpha_loss = -tf.reduce_mean(alpha_loss)

        grads = tape.gradient(alpha_loss, self.log_alpha)
        self.alpha_optimizer.apply_gradients(zip([grads], [self.log_alpha]))

        # debug
        self.log(alpha_loss=alpha_loss, alpha_target=target_alpha, alpha_gradient=grads)
