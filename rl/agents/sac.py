"""Soft Actor-Critic (SAC) Agent"""

import os
import gym
import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from typing import Union, List, Dict, Tuple

from rl import utils
from rl.agents import Agent
from rl.agents.dqn import ReplayMemory
from rl.parameters import DynamicParameter, LearnableParameter
from rl.networks.networks import Network

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import losses
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class SACAgent(Agent):
    # TODO: add noise-action?
    # TODO: make `polyak`, `optimization_steps`, `memory`, `discount`, and `clip_norm` DynamicParameters?
    def __init__(self, *args, optimizer='adam', actor_lr: Union[float, LearningRateSchedule, DynamicParameter] = 3e-3,
                 critic_lr: Union[float, LearningRateSchedule, DynamicParameter] = 3e-4, name='sac-agent', gamma=0.99,
                 load=False, clip_norm: Union[None, float, Tuple[float, float]] = None, polyak=0.995, repeat_action=1,
                 memory_size=1000, optimization_steps=1, critic: dict = None, actor: dict = None,
                 target_entropy: Union[None, str, float] = None,
                 temperature: Union[float, str, LearningRateSchedule, DynamicParameter] = 0.2, **kwargs):
        assert 0.0 < polyak <= 1.0
        assert repeat_action >= 1
        assert optimization_steps >= 1

        super().__init__(*args, name=name, **kwargs)
        assert memory_size > self.batch_size * optimization_steps

        self.gamma = gamma
        self.repeat_action = repeat_action
        self.drop_batch_remainder = memory_size % self.batch_size != 0

        # Action space
        self._init_action_space()

        # Temperature parameter (alpha)
        if target_entropy is not None:
            self.should_adjust_entropy = True

            if isinstance(target_entropy, float):
                self.target_entropy = tf.constant(target_entropy, dtype=tf.float32)
            else:
                # 'auto' => Target entropy is -|A|
                self.target_entropy = tf.constant(-np.prod(self.action_shape), dtype=tf.float32)

            if temperature == 'auto':
                self.temperature = LearnableParameter(initial_value=1.0)
                # self.temperature = DynamicParameter.create(value=1.0)
                self.log_temperature = LearnableParameter(initial_value=0.0)  # 0 = log(1)
            else:
                assert isinstance(temperature, float)
                assert temperature > 0.0

                self.temperature = LearnableParameter(initial_value=temperature)
                # self.temperature = DynamicParameter.create(value=temperature)
                self.log_temperature = LearnableParameter(initial_value=tf.math.log(temperature))
        else:
            assert not isinstance(temperature, str)

            self.should_adjust_entropy = False
            self.temperature = DynamicParameter.create(value=temperature)

        # Memory
        self.memory = self.get_memory(size=memory_size)

        # Gradient clipping
        self._init_gradient_clipping(clip_norm)

        # Networks (and target-networks)
        self.weights_path = dict(policy=os.path.join(self.base_path, 'policy'),
                                 q_net1=os.path.join(self.base_path, 'q_net1'),
                                 q_net2=os.path.join(self.base_path, 'q_net2'))

        critic = {} if critic is None else critic
        actor = {} if actor is None else actor

        self.policy = PolicyNetwork(agent=self, **actor)
        self.q_net1 = QNetwork(agent=self, weights_path=self.weights_path['q_net1'], **critic)
        self.q_net2 = QNetwork(agent=self, weights_path=self.weights_path['q_net2'], **critic)

        self.target_q1 = QNetwork(agent=self, weights_path='', **critic)
        self.target_q2 = QNetwork(agent=self, weights_path='', **critic)

        self.target_q1.set_weights(weights=self.q_net1.get_weights())
        self.target_q2.set_weights(weights=self.q_net2.get_weights())

        # Optimization
        self.actor_lr = DynamicParameter.create(value=actor_lr)
        self.critic_lr = DynamicParameter.create(value=critic_lr)

        self.policy_optimizer = utils.get_optimizer_by_name(optimizer, learning_rate=self.actor_lr)
        self.q1_optimizer = utils.get_optimizer_by_name(optimizer, learning_rate=self.critic_lr)
        self.q2_optimizer = utils.get_optimizer_by_name(optimizer, learning_rate=self.critic_lr)
        self.optimization_steps = optimization_steps
        self.polyak_coeff = polyak

        if self.should_adjust_entropy:
            self.entropy_lr = DynamicParameter.create(value=actor_lr)  # use same lr as actor
            self.entropy_optimizer = utils.get_optimizer_by_name(optimizer, learning_rate=self.entropy_lr)

        if load:
            self.load()

    def _init_gradient_clipping(self, clip_norm: Union[float, tuple, None]):
        if clip_norm is None:
            self.should_clip_policy_grads = False
            self.should_clip_q_net_grads = False

        elif isinstance(clip_norm, tuple):
            assert len(clip_norm) == 2

            if isinstance(clip_norm[0], float):
                assert clip_norm[0] > 0.0

                self.should_clip_policy_grads = True
                self.policy_clip_norm = tf.constant(clip_norm[0], dtype=tf.float32)
            else:
                self.should_clip_policy_grads = False

            if isinstance(clip_norm[1], float):
                assert clip_norm[1] > 0.0

                self.should_clip_q_net_grads = True
                self.q_net_clip_norm = tf.constant(clip_norm[1], dtype=tf.float32)
            else:
                self.should_clip_q_net_grads = False

        elif isinstance(clip_norm, float):
            assert clip_norm > 0.0

            self.should_clip_policy_grads = True
            self.should_clip_q_net_grads = True

            self.policy_clip_norm = tf.constant(clip_norm, dtype=tf.float32)
            self.q_net_clip_norm = tf.constant(clip_norm, dtype=tf.float32)
        else:
            raise TypeError(f'`clip_norm` should be "None", "float", or "tuple" not "{type(clip_norm)}"!')

    def _init_action_space(self):
        action_space = self.env.action_space

        if isinstance(action_space, gym.spaces.Box):
            self.num_actions = action_space.shape[0]
            self.action_shape = action_space.shape

            # continuous:
            if action_space.is_bounded():
                self.distribution_type = 'beta'

                self.action_low = tf.constant(action_space.low, dtype=tf.float32)
                self.action_high = tf.constant(action_space.high, dtype=tf.float32)
                self.action_range = tf.constant(action_space.high - action_space.low,
                                                dtype=tf.float32)

                self.convert_action = lambda a: tf.squeeze(a * self.action_range + self.action_low).numpy()
            else:
                self.distribution_type = 'gaussian'
                self.convert_action = lambda a: a[0].numpy()
        else:
            # discrete:
            assert not isinstance(action_space, gym.spaces.MultiDiscrete)

            self.distribution_type = 'categorical'
            self.num_actions = 1
            self.num_classes = action_space.n
            self.action_shape = (self.num_classes - 1,)

            def convert_action(a):
                a = tf.round(a)
                return tf.cast(tf.squeeze(a), dtype=tf.int32).numpy()

            self.convert_action = convert_action

    def _init_action_space2(self):
        action_space = self.env.action_space

        if isinstance(action_space, gym.spaces.Box):
            self.num_actions = action_space.shape[0]
            self.action_shape = action_space.shape

            # continuous:
            if action_space.is_bounded():
                self.distribution_type = 'beta'

                self.action_low = tf.constant(action_space.low, dtype=tf.float32)
                self.action_high = tf.constant(action_space.high, dtype=tf.float32)
                self.action_range = tf.constant(action_space.high - action_space.low,
                                                dtype=tf.float32)

                self.convert_action = lambda a: tf.squeeze(a * self.action_range + self.action_low).numpy()
            else:
                self.distribution_type = 'gaussian'
                self.convert_action = lambda a: a[0].numpy()
        else:
            # discrete:
            self.distribution_type = 'beta'
            # self.distribution_type = 'gaussian'
            self.num_actions = 1
            self.num_classes = action_space.n
            self.action_range = self.num_classes - 1
            self.action_shape = (self.action_range,)

            def convert_action(a):
                a = tf.squeeze(a * self.action_range)
                a = tf.round(a)
                # a = tf.clip_by_value(a, clip_value_min=0.0, clip_value_max=self.action_range)
                return tf.cast(a, dtype=tf.int32).numpy()

            self.convert_action = convert_action

    def get_action(self, states):
        actions = self.policy.predict(inputs=states, training=False)[:1]
        return actions

    def get_memory(self, size: int):
        return ReplayMemory(state_spec=self.state_spec, num_actions=self.num_actions, size=size)

    def update(self):
        if len(self.memory) < self.batch_size * self.optimization_steps:
            print('Not updated: few experience in memory!')
            return

        t0 = time.time()

        for _ in range(self.optimization_steps):
            batch = self.memory.sample_batch(batch_size=self.batch_size, seed=self.seed)

            # Update Q-Networks:
            grads_q1, grads_q2 = self.get_q_networks_gradients(batch)
            applied_grads_q = self.apply_q_network_gradients(gradients=[grads_q1, grads_q2])

            # Update policy:
            policy_grads = self.get_policy_gradients(batch)
            applied_grads_policy = self.apply_policy_gradients(gradients=policy_grads)

            self.update_target_networks()

            # Update temperature (alpha) parameter:
            if self.should_adjust_entropy:
                with tf.GradientTape() as tape:
                    entropy_loss = self.entropy_objective(batch)

                # entropy_grads = tape.gradient(entropy_loss, self.temperature.variable)
                # self.entropy_optimizer.apply_gradients(zip(entropy_grads, self.temperature.variable))
                entropy_grads = tape.gradient(entropy_loss, self.log_temperature.variable)
                self.entropy_optimizer.apply_gradients(zip(entropy_grads, self.log_temperature.variable))
                self.temperature.value = tf.exp(self.log_temperature.value)

                self.log(gradient_value_temperature=[g for g in entropy_grads],
                         loss_entropy=entropy_loss, temperature_log=self.log_temperature.value)

            # debug
            self.log(gradients_policy=[tf.norm(g) for g in policy_grads], actor_lr=self.actor_lr.value,
                     gradients_policy_applied=[tf.norm(g) for g in applied_grads_policy],
                     gradients_q1=[tf.norm(g) for g in grads_q1], gradients_q2=[tf.norm(g) for g in grads_q2],
                     gradients_q1_applied=[tf.norm(g) for g in applied_grads_q[0]], critic_lr=self.critic_lr.value,
                     gradients_q2_applied=[tf.norm(g) for g in applied_grads_q[1]])

        print(f'Update took {round(time.time() - t0, 3)}s.')

    def update_target_networks(self):
        """Updates the weights of the target networks by Polyak average"""
        self.polyak_averaging(model=self.q_net1, target=self.target_q1)
        self.polyak_averaging(model=self.q_net2, target=self.target_q2)

    def polyak_averaging(self, model: Network, target: Network):
        for var, var_target in zip(model.trainable_variables(), target.trainable_variables()):
            value = self.polyak_coeff * var_target + (1.0 - self.polyak_coeff) * var
            var_target.assign(value, read_value=False)

    def get_policy_gradients(self, batch):
        with tf.GradientTape() as tape:
            loss_policy = self.policy_objective(batch)

        gradients = tape.gradient(loss_policy, self.policy.trainable_variables())
        return gradients

    def get_q_networks_gradients(self, batch):
        with tf.GradientTape(persistent=True) as tape:
            loss_q1, loss_q2 = self.q_network_objective(batch)

        gradients_q1 = tape.gradient(loss_q1, self.q_net1.trainable_variables())
        gradients_q2 = tape.gradient(loss_q2, self.q_net2.trainable_variables())
        del tape
        return gradients_q1, gradients_q2

    def get_temperature_gradients(self, batch):
        with tf.GradientTape() as tape:
            loss = self.entropy_objective(batch)

        gradients = tape.gradient(loss, self.temperature.variable)
        return gradients

    def apply_policy_gradients(self, gradients) -> List[tf.Tensor]:
        if self.should_clip_policy_grads:
            gradients = utils.clip_gradients(gradients, norm=self.policy_clip_norm)

        self.policy_optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables()))
        return gradients

    def apply_q_network_gradients(self, gradients: list) -> List[tf.Tensor]:
        if self.should_clip_q_net_grads:
            gradients[0] = utils.clip_gradients(gradients[0], norm=self.q_net_clip_norm)
            gradients[1] = utils.clip_gradients(gradients[1], norm=self.q_net_clip_norm)

        self.q1_optimizer.apply_gradients(zip(gradients[0], self.q_net1.trainable_variables()))
        self.q2_optimizer.apply_gradients(zip(gradients[1], self.q_net2.trainable_variables()))
        return gradients

    # @tf.function
    # def policy_objective(self, batch):
    #     states = batch[0]
    #     actions, log_prob, mean, std = self.policy.predict(inputs=states)
    #
    #     # predict Q-values from both Q-functions and then take the minimum of the two
    #     q_values1 = self.q_net1.predict(states, actions, training=True)
    #     q_values2 = self.q_net2.predict(states, actions, training=True)
    #     q_values = tf.minimum(q_values1, q_values2)
    #
    #     # loss = q_values - self.temperature() * tf.exp(log_prob)
    #     loss = q_values - self.temperature() * log_prob
    #
    #     # debug
    #     self.log(loss_policy=loss, actions_policy=actions, q_values1=q_values1, q_values2=q_values2,
    #              q_values=q_values, distribution_mean=mean, distribution_std=std, log_prob=log_prob,
    #              temperature=self.temperature.value, entropy_policy=self.temperature.value * log_prob)
    #
    #     # minus sign '-' for ascent direction
    #     return -loss

    def policy_objective(self, batch):
        states = batch[0]
        actions, log_prob, mean, std = self.policy.predict(inputs=states)

        # predict Q-values from both Q-functions and then take the minimum of the two
        q_values1 = self.q_net1.predict(states, actions, training=True)
        q_values2 = self.q_net2.predict(states, actions, training=True)

        q_values1 = self.combine(q_values1)
        q_values2 = self.combine(q_values2)
        q_values = tf.expand_dims(tf.minimum(q_values1, q_values2), axis=1)

        # loss = q_values - self.temperature() * tf.exp(log_prob)
        entropy = self.temperature() * tf.reduce_mean(log_prob, axis=1, keepdims=True)
        # entropy = self.temperature() * log_prob

        # loss = q_values - entropy
        loss = tf.reduce_mean(entropy - q_values)

        # debug
        self.log(average=True, loss_policy=loss, actions_policy=actions, log_prob=log_prob,
                 q_values1=q_values1, q_values2=q_values2,
                 q_values=q_values, distribution_mean=mean, distribution_std=std,
                 temperature=self.temperature.value, entropy_policy=entropy)

        return loss

    @staticmethod
    def combine(q):
        base = q[:, 0]
        exp = q[:, 1]
        return base * tf.pow(10.0, exp)

    @staticmethod
    def decompose(tensor):
        base, exp = tf.map_fn(fn=utils.decompose_number, elems=tensor, dtype=(tf.float32, tf.float32))
        return tf.concat([base, tf.expand_dims(exp, 1)], axis=1)

    # @tf.function
    def q_network_objective(self, batch):
        states, actions, rewards, next_states, terminals = batch
        next_actions, log_prob, mean, std = self.policy.predict(next_states)

        # compute targets
        target_q_values1 = self.target_q1.predict(next_states, next_actions, training=True)
        target_q_values2 = self.target_q2.predict(next_states, next_actions, training=True)
        # target_q_values = tf.minimum(target_q_values1, target_q_values2)
        target_q_values = tf.minimum(x=self.combine(target_q_values1),
                                     y=self.combine(target_q_values2))
        target_q_values = tf.expand_dims(target_q_values, axis=1)

        entropy = self.temperature() * tf.reduce_mean(log_prob, axis=1, keepdims=True)

        # targets = rewards * self.gamma * (1.0 - terminals) * (target_q_values - self.temperature() * log_prob)
        targets = tf.stop_gradient(
            rewards + self.gamma * (1.0 - terminals) * (target_q_values - entropy))

        # compute two losses, for the two Q-networks respectively
        q_values1 = self.q_net1.predict(states, actions, training=True)
        q_values2 = self.q_net2.predict(states, actions, training=True)

        # loss_q1 = 0.5 * losses.MSE(q_values1, targets)
        # loss_q2 = 0.5 * losses.MSE(q_values2, targets)
        loss_q1 = self.q_loss(q_values1, targets=self.decompose(targets))
        loss_q2 = self.q_loss(q_values2, targets=self.decompose(targets))

        # debug
        self.log(average=True, loss_q1=loss_q1, loss_q2=loss_q2, actions_q=next_actions, targets=targets,
                 distribution_std_q=std, q_values1=self.combine(q_values1), q_values2=self.combine(q_values2),
                 target_q_values1=self.combine(target_q_values1), target_q_values2=self.combine(target_q_values2),
                 target_q_values=target_q_values, entropy_q=entropy, terminals=terminals,
                 temperature=self.temperature.value, log_prob_q=log_prob, distribution_mean_q=mean)

        return loss_q1, loss_q2

    def q_loss(self, q_values, targets):
        base_loss = tf.reduce_mean((q_values[:, 0] - targets[:, 0]) ** 2)
        exp_loss = tf.reduce_mean((q_values[:, 1] - targets[:, 1]) ** 2)

        return 0.5 * (0.25 * base_loss + exp_loss / (self.q_net1.exp_scale ** 2))

    def entropy_objective(self, batch):
        states = batch[0]
        log_prob = self.policy.predict(inputs=states)[1]

        loss = -tf.reduce_mean(
            self.log_temperature.value * tf.stop_gradient(log_prob + self.target_entropy))

        # loss = tf.reduce_mean(
        #     -self.temperature.value * tf.stop_gradient(log_prob + self.target_entropy))

        return loss
        # return loss, dict(loss_entropy=loss, temperature_log=self.log_temperature.value)

    def learn(self, episodes: int, timesteps: int, save_every: Union[bool, str, int] = False,
              render_every: Union[bool, str, int] = False, close=True):
        if save_every is False:
            save_every = episodes + 1
        elif save_every is True:
            save_every = 1
        elif save_every == 'end':
            save_every = episodes
        else:
            assert episodes % save_every == 0

        if render_every is False:
            render_every = episodes + 1
        elif render_every is True:
            render_every = 1

        try:
            for episode in range(1, episodes + 1):
                preprocess_fn = self.preprocess()
                self.reset()

                state = self.env.reset()
                state = preprocess_fn(state)
                state = utils.to_tensor(state)

                episode_reward = 0.0
                t0 = time.time()
                render = episode % render_every == 0

                for t in range(1, timesteps + 1):
                    if render:
                        self.env.render()

                    if isinstance(state, dict):
                        state = {f'state_{k}': v for k, v in state.items()}

                    # Agent prediction
                    action = self.get_action(state)
                    action_env = self.convert_action(action)

                    # Environment step
                    for _ in range(self.repeat_action):
                        next_state, reward, terminal, _ = self.env.step(action_env)
                        episode_reward += reward

                        if terminal:
                            break

                    self.log(actions=action, action_env=action_env, rewards=reward)

                    next_state = preprocess_fn(next_state)
                    next_state = utils.to_tensor(next_state)

                    self.memory.append(state, action, reward, next_state, terminal)
                    state = next_state

                    # check whether a termination (terminal state or end of a transition) is reached:
                    if terminal or (t == timesteps):
                        print(f'Episode {episode} terminated after {t} timesteps in {round((time.time() - t0), 3)}s ' +
                              f'with reward {round(episode_reward, 3)}.')

                        self.update()
                        break

                self.memory.ensure_space()

                # Logging
                self.log(episode_rewards=episode_reward)
                self.write_summaries()

                if self.should_record:
                    self.record(episode)

                self.on_episode_end()

                if episode % save_every == 0:
                    self.save()
        finally:
            if close:
                print('closing...')
                self.env.close()

    def on_episode_end(self):
        self.temperature.on_episode()

    def load_weights(self):
        print('[SAC] loading weights...')
        self.policy.load_weights()
        self.q_net1.load_weights()
        self.q_net1.load_weights()

        self.target_q1.set_weights(weights=self.q_net1.get_weights())
        self.target_q2.set_weights(weights=self.q_net2.get_weights())

    def save_weights(self):
        self.policy.save_weights()
        self.q_net1.save_weights()
        self.q_net2.save_weights()

    def summary(self):
        self.policy.summary()
        self.q_net1.summary()


class PolicyNetwork(Network):
    def __init__(self, agent: SACAgent, **kwargs):
        super().__init__(agent=agent)

        self.distribution = self.agent.distribution_type
        self.weights_path = self.agent.weights_path['policy']
        self.net = self.build(**kwargs)

    @tf.function
    def predict(self, inputs: Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]], training=False):
        """Samples actions, and outputs their log-probabilities as well"""
        policy: tfp.distributions.Distribution = self.net(inputs, training=training)
        actions = policy

        if self.distribution != 'categorical':
            log_prob = policy.log_prob(self._clip_actions(actions=policy))
            mean = policy.mean()
            std = policy.stddev()
        else:
            mean = std = 0.0
            log_prob = policy.log_prob(policy)
            actions = tf.reshape(actions, shape=(actions.shape[1], actions.shape[0]))

        # mean, std are useful for debugging
        return actions, tf.reshape(log_prob, actions.shape), mean, std

    def build(self, **kwargs) -> Model:
        dist_args = kwargs.pop('distribution', dict(bias_initializer='zeros', kernel_initializer='glorot_uniform'))

        inputs = self._get_input_layers()
        last_layer = self.layers(inputs, **kwargs)
        actions = self.get_distribution_layer(distribution=self.distribution, layer=last_layer, **dist_args)

        return Model(inputs, outputs=actions, name='Policy-Network')

    def layers(self, inputs: Dict[str, Input], **kwargs) -> Layer:
        units = kwargs.get('units', 64)
        num_layers = kwargs.get('num_layers', kwargs.get('layers', 2))  # 'num_layers' or 'layers'
        dropout_rate = kwargs.get('dropout', 0.0)
        normalization = kwargs.get('normalization', 'layer')
        args = dict(activation=kwargs.get('activation', tf.nn.tanh),
                    bias_initializer=kwargs.get('bias_initializer', 'glorot_uniform'),
                    kernel_initializer=kwargs.get('kernel_initializer', 'glorot_normal'))

        x = Dense(units, **args)(inputs['state'])
        x = utils.normalization_layer(x, name=normalization)

        for _ in range(num_layers):
            if dropout_rate > 0.0:
                x = Dense(units, **args)(x)
                x = Dropout(rate=dropout_rate)(x)
            else:
                x = Dense(units, **args)(x)

            x = utils.normalization_layer(x, name=normalization)

        return x

    def trainable_variables(self):
        return self.net.trainable_variables

    def load_weights(self):
        self.net.load_weights(filepath=self.weights_path, by_name=False)

    def save_weights(self):
        self.net.save_weights(filepath=self.weights_path)

    def summary(self):
        print('==== [SAC] Policy Network ====')
        self.net.summary()


class QNetwork(Network):
    def __init__(self, agent: SACAgent, weights_path: str, **kwargs):
        super().__init__(agent=agent)

        self.discrete_action_space = self.agent.distribution_type == 'categorical'
        self.weights_path = weights_path

        self.exp_scale = None
        self.net = self.build(**kwargs)

    # @tf.function
    # def predict(self, states: Union[tf.Tensor, List[tf.Tensor]], actions: Union[tf.Tensor, List[tf.Tensor]],
    #             training=False):
    #     q_values = self.net(inputs=dict(state=states, action=actions), training=training)
    #     tf.print(q_values.shape)
    #     breakpoint()
    #
    #     if self.discrete_action_space:
    #         # index |A|-dimensional q-values by given `actions`
    #         num_classes = self.agent.num_classes
    #         shape = (q_values.shape[0], 1)
    #
    #         q_indices = tf.concat([
    #             tf.reshape(tf.range(start=0, limit=shape[0], dtype=tf.int32), shape),
    #             tf.cast(actions, dtype=tf.int32)
    #         ], axis=1)
    #
    #         tf.print(q_indices.shape)
    #         breakpoint()
    #
    #         # select 1 q-value per action (select corresponding bases and exponents)
    #         selected_q_values = tf.gather_nd(q_values, q_indices)
    #         tf.print(selected_q_values.shape)
    #         breakpoint()
    #
    #         # TODO: expand_dims
    #         return selected_q_values
    #
    #     return q_values

    @tf.function
    def predict(self, states: Union[tf.Tensor, List[tf.Tensor]], actions: Union[tf.Tensor, List[tf.Tensor]],
                training=False):
        q_values = self.net(inputs=dict(state=states, action=actions), training=training)

        if self.discrete_action_space:
            # index |A|-dimensional q-values by given `actions`
            num_classes = self.agent.num_classes
            shape = (q_values.shape[0], 1)

            q_indices = tf.concat([
                tf.reshape(tf.range(start=0, limit=shape[0], dtype=tf.int32), shape),
                tf.cast(actions, dtype=tf.int32)
            ], axis=1)

            # select 1 q-value per action (select corresponding bases and exponents)
            selected_bases = tf.gather_nd(q_values[:, :num_classes], q_indices)
            selected_exp = tf.gather_nd(q_values[:, num_classes:], q_indices)

            return tf.stack([selected_bases, selected_exp], axis=1)

        return q_values

    def build(self, **kwargs) -> Model:
        self.exp_scale = tf.constant(kwargs.pop('exponent_scale', 3.0), dtype=tf.float32)
        q_args = kwargs.pop('output', dict(bias_initializer='zeros', kernel_initializer='glorot_uniform'))

        inputs = self._get_input_layers()
        inputs['action'] = Input(shape=self.agent.action_shape, dtype=tf.float32, name='action')

        last_layer = self.layers(inputs, **kwargs)
        q_values = self.output_layer(last_layer, **q_args)

        return Model(inputs, outputs=q_values, name='Q-Network')

    def layers(self, inputs: Dict[str, Input], **kwargs) -> Layer:
        units = kwargs.get('units', 64)
        units_action = kwargs.get('units_action', 16)
        num_layers = kwargs.get('num_layers', kwargs.get('layers', 2))  # 'num_layers' or 'layers'
        dropout_rate = kwargs.get('dropout', 0.0)
        normalization = kwargs.get('normalization', 'layer')

        args = dict(activation=kwargs.get('activation', tf.nn.swish),
                    bias_initializer=kwargs.get('bias_initializer', 'glorot_uniform'),
                    kernel_initializer=kwargs.get('kernel_initializer', 'glorot_normal'))

        state_branch = self._branch(inputs['state'], units, num_layers, dropout_rate, normalization, **args)
        action_branch = self._branch(inputs['action'], units_action, num_layers, dropout_rate, normalization, **args)

        x = Dense(units, **args)(concatenate([state_branch, action_branch]))
        x = utils.normalization_layer(x, name=normalization)
        return x

    # def output_layer(self, layer: Layer, **kwargs) -> Layer:
    #     if self.discrete_action_space:
    #         assert self.agent.num_actions == 1
    #         num_classes = self.agent.num_classes
    #
    #         # # q_values = q_bases * 10^q_exp
    #         q_base = Dense(units=num_classes, activation=tf.nn.tanh, name='q-base', **kwargs)(layer)
    #         q_exp = Dense(units=num_classes, activation=lambda x: self.exp_scale * tf.nn.sigmoid(x), name='q-exponent',
    #                       **kwargs)(layer)
    #
    #         # q_values = Dense(units=self.agent.num_classes, bias_initializer='glorot_uniform',
    #         #                  name='discrete-q-values')(layer)
    #         # return q_values
    #         return concatenate([q_base, q_exp], axis=1)
    #
    #         # # index |A|-dimensional q-values by input `actions`
    #         # shape = (q_values.shape[0], 1)
    #         #
    #         # q_indices = tf.concat([
    #         #     tf.reshape(tf.range(start=0, limit=shape[0], dtype=tf.int32), shape),
    #         #     tf.cast(inputs['action'], dtype=tf.int32)
    #         # ], axis=1)
    #         #
    #         # # select 1 q-value per action
    #         # selected_q_values = tf.gather_nd(q_values, q_indices)
    #         # # selected_q_values.set_shape(shape)
    #         # return selected_q_values
    #     else:
    #         # q_values = Dense(units=1, bias_initializer='glorot_uniform', name='continuous-q-values')(layer)
    #         q_base = Dense(units=1, activation=tf.nn.tanh, name='q-base', **kwargs)(layer)
    #         q_exp = Dense(units=1, activation=lambda x: self.exp_scale * tf.nn.sigmoid(x), name='q-exponent',
    #                       **kwargs)(layer)
    #
    #         return concatenate([q_base, q_exp], axis=1)

    def output_layer(self, layer: Layer, **kwargs) -> Layer:
        if self.discrete_action_space:
            num_classes = self.agent.num_classes

            # # q_values = q_bases * 10^q_exp
            q_base = Dense(units=num_classes, activation=tf.nn.tanh, name='q-base', **kwargs)(layer)
            q_exp = Dense(units=num_classes, activation=lambda x: self.exp_scale * tf.nn.sigmoid(x), name='q-exponent',
                          **kwargs)(layer)

            return concatenate([q_base, q_exp], axis=1)
        else:
            q_base = Dense(units=1, activation=tf.nn.tanh, name='q-base', **kwargs)(layer)
            q_exp = Dense(units=1, activation=lambda x: self.exp_scale * tf.nn.sigmoid(x), name='q-exponent',
                          **kwargs)(layer)

            return concatenate([q_base, q_exp], axis=1)

    def _branch(self, input_layer: Input, units: int, num_layers: int, dropout_rate: float,
                normalization: str, **kwargs) -> Layer:
        x = Dense(units, **kwargs)(input_layer)
        x = utils.normalization_layer(x, name=normalization)

        for _ in range(num_layers):
            if dropout_rate > 0.0:
                x = Dense(units, **kwargs)(x)
                x = Dropout(rate=dropout_rate)(x)
            else:
                x = Dense(units, **kwargs)(x)

            x = utils.normalization_layer(x, name=normalization)

        return x

    def trainable_variables(self):
        return self.net.trainable_variables

    def get_weights(self):
        return self.net.get_weights()

    def set_weights(self, weights):
        self.net.set_weights(weights)

    def load_weights(self):
        self.net.load_weights(filepath=self.weights_path, by_name=False)

    def save_weights(self):
        self.net.save_weights(filepath=self.weights_path)

    def summary(self):
        print('==== [SAC] Q-Network ====')
        self.net.summary()
