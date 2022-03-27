"""Deep Deterministic Policy Gradient (DDPG)
    - Continuous control with deep reinforcement learning (arXiv:1509.02971)
"""

import tensorflow as tf

from tensorflow.keras.layers import Layer, Dense, Concatenate, Input
from typing import Dict, Tuple, Union

from rl import utils
from rl.agents import Agent
from rl.parameters import DynamicParameter
from rl.agents.actions import TanhConverter
from rl.memories import TransitionSpec, ReplayMemory, PrioritizedMemory
from rl.networks import backbones, Network, DeterministicPolicyNetwork


@Network.register(name='DDPG-ActorNetwork')
class ActorNetwork(DeterministicPolicyNetwork):

    @tf.function
    def objective(self, batch, reduction=tf.reduce_mean) -> tuple:
        states = batch['state']
        actions = self(states, training=True)

        q_values = self.agent.critic((states, actions), training=False)
        loss = -reduction(q_values)

        return loss, dict(loss=loss, actions=actions, q_values=q_values)


@Network.register(name='DDPG-CriticNetwork')
class CriticNetwork(Network):

    def call(self, *inputs, training=None, **kwargs):
        return super().call(inputs, training=training, **kwargs)

    def structure(self, inputs: Dict[str, Input], **kwargs) -> tuple:
        state_in = inputs['state']
        action_in = inputs['action']

        # preprocessing
        preproc_in = self.apply_preprocessing(inputs, preprocess=kwargs.pop('preprocess', None))

        x = Concatenate()([preproc_in['state'], preproc_in['action']])
        x = backbones.dense(x, **kwargs)

        out = self.output_layer(x, **self.output_kwargs)
        return (state_in, action_in), out

    def output_layer(self, layer: Layer, **kwargs) -> Layer:
        return Dense(units=self.agent.num_actions, name='q-values', **kwargs)(layer)

    def get_inputs(self) -> Dict[str, Input]:
        inputs = super().get_inputs()
        inputs['action'] = Input(shape=(self.agent.num_actions,), name='action', dtype=tf.float32)
        return inputs

    @tf.function
    def objective(self, batch: dict, reduction=tf.reduce_mean) -> tuple:
        q_values = self((batch['state'], batch['action']), training=True)
        q_targets, debug = self.targets(batch)
        td_error = q_values - q_targets

        if self.agent.prioritized:
            # inform agent's memory about td-error, to later update priorities;
            # compared to DQN, we take the mean over the action-axis
            self.agent.memory.td_error.assign(tf.stop_gradient(tf.reduce_mean(td_error, axis=-1)))

            loss = 0.5 * reduction(tf.square(td_error * batch['_weights']))
        else:
            loss = 0.5 * reduction(tf.square(td_error))

        debug.update(loss=loss, q_values=q_values, td_error=td_error)

        if '_weights' in batch:
            debug['weights_IS'] = batch['_weights']
            debug['td_error_weighted'] = tf.stop_gradient(td_error * batch['_weights'])

        return loss, debug

    @tf.function
    def targets(self, batch: dict):
        next_states = batch['next_state']

        argmax_a = self.agent.actor.target(next_states, training=False)
        q_values = self.target((next_states, argmax_a), training=False)

        targets = batch['reward'] + self.agent.gamma * q_values * (1.0 - batch['terminal'])
        targets = tf.stop_gradient(targets)

        return targets, dict(next_actions=argmax_a, next_q_values=q_values, targets=targets)


# TODO: support for discrete actions?
class DDPG(Agent):

    def __init__(self, *args, name='ddpg', actor_lr: utils.DynamicType = 1e-4, critic_lr: utils.DynamicType = 1e-3,
                 optimizer='adam', actor: dict = None, critic: dict = None, clip_norm=(None, None), polyak=0.95,
                 memory_size=1024, noise: utils.DynamicType = 0, prioritized=False,
                 alpha: utils.DynamicType = 0.6, beta: utils.DynamicType = 0.1, **kwargs):
        assert memory_size >= 1
        assert 0.0 < polyak <= 1.0

        super().__init__(*args, name=name, **kwargs)
        self.num_actions = self.action_converter.num_actions

        # hyper-parameters
        self.memory_size = int(memory_size)
        self.critic_lr = DynamicParameter.create(value=critic_lr)
        self.actor_lr = DynamicParameter.create(value=actor_lr)
        self.polyak = float(polyak)
        self.noise = DynamicParameter.create(value=noise)
        self.prioritized = bool(prioritized)

        # PER memory params:
        if self.prioritized:
            self.alpha = DynamicParameter.create(value=alpha)
            self.beta = DynamicParameter.create(value=beta)

        # Networks
        self.actor = Network.create(agent=self, target=True, log_prefix='actor', **(actor or {}),
                                    base_class=ActorNetwork)

        self.critic = Network.create(agent=self, target=True, log_prefix='critic', **(critic or {}),
                                     base_class=CriticNetwork)

        self.actor.compile(optimizer, clip_norm=clip_norm[0], clip=self.clip_grads, learning_rate=self.actor_lr)
        self.critic.compile(optimizer, clip_norm=clip_norm[1], clip=self.clip_grads, learning_rate=self.critic_lr)

    @property
    def transition_spec(self) -> TransitionSpec:
        return TransitionSpec(state=self.state_spec, next_state=True, action=self.num_actions, terminal=True)

    def define_memory(self) -> Union[ReplayMemory, PrioritizedMemory]:
        if self.prioritized:
            return PrioritizedMemory(self.transition_spec, shape=self.memory_size, gamma=self.gamma,
                                     alpha=self.alpha, beta=self.beta, seed=self.seed)

        return ReplayMemory(self.transition_spec, shape=self.memory_size, seed=self.seed)

    def define_action_converter(self, kwargs: dict) -> TanhConverter:
        return TanhConverter(space=self.env.action_space, **(kwargs or {}))

    def act(self, state, deterministic=False, **kwargs) -> Tuple[tf.Tensor, dict, dict]:
        greedy_action = self.actor(state, **kwargs)
        debug = {}

        if (not deterministic) and (self.noise.value > 0.0):
            # add random noise for exploration
            noise = tf.random.normal(shape=greedy_action.shape, stddev=self.noise(), seed=self.seed)
            action = tf.clip_by_value(greedy_action + noise, clip_value_min=-1.0, clip_value_max=1.0)

            debug.update(noise=noise, noise_std=self.noise.value, noise_ratio=self._noise_ratio(greedy_action, action))
            return action, {}, debug

        return greedy_action, {}, debug

    def _noise_ratio(self, greedy_action, action):
        return tf.reduce_mean(tf.abs((greedy_action - action) / self.action_converter.action_range))

    @tf.function
    def act_randomly(self, state) -> Tuple[tf.Tensor, dict, dict]:
        action = self.actor(state)

        # add random noise for exploration
        noise = tf.random.normal(shape=action.shape, stddev=self.noise(), seed=self.seed)
        action = tf.clip_by_value(action + noise, clip_value_min=-1.0, clip_value_max=1.0)

        return action, {}, {}

    def learn(self, *args, **kwargs):
        with utils.Timed('Learn'):
            super().learn(*args, **kwargs)

    def update(self):
        batch = self.memory.get_batch(batch_size=self.batch_size)

        self.critic.train_step(batch)
        self.actor.train_step(batch)

        self.update_target_networks()

    def update_target_networks(self):
        self.critic.update_target_network(polyak=self.polyak)
        self.actor.update_target_network(polyak=self.polyak)

        self.log(target_actor_distance=self.actor.debug_target_network(),
                 target_critic_distance=self.critic.debug_target_network())

    def on_transition(self, *args, exploration=False):
        super().on_transition(*args, exploration=exploration)

        if not exploration:
            self.update()
