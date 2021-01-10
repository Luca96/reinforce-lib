"""Deep Deterministic Policy Gradient (DDPG) Agent"""

import os
import gym
import time
import numpy as np
import tensorflow as tf

from typing import Union, List, Dict, Tuple

from rl import utils
from rl.agents import Agent
from rl.agents.dqn import ReplayMemory
from rl.parameters import DynamicParameter
from rl.networks.networks import Network

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import losses
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class DDPGAgent(Agent):
    # TODO: allow different noise process
    # TODO: add L2-weight decay for critic (l2=10e-2)
    # TODO: network customization
    def __init__(self, *args, optimizer='adam', actor_lr: Union[float, LearningRateSchedule, DynamicParameter] = 1e-4,
                 critic_lr: Union[float, LearningRateSchedule, DynamicParameter] = 1e-3, name='ddpg-agent', gamma=0.99,
                 load=False, clip_norm: Union[None, float, Tuple[float, float]] = None, polyak=0.999, repeat_action=1,
                 memory_size=1000, noise: Union[float, LearningRateSchedule, DynamicParameter] = 0.005,
                 optimization_steps=1, critic: dict = None, actor: dict = None, **kwargs):
        assert 0.0 < polyak <= 1.0
        assert repeat_action >= 1
        assert optimization_steps >= 1

        super().__init__(*args, name=name, **kwargs)
        assert memory_size > self.batch_size

        self.gamma = gamma
        self.repeat_action = repeat_action
        self.drop_batch_remainder = memory_size % self.batch_size != 0

        # Noise process (Gaussian Noise)
        if isinstance(noise, float):
            assert noise >= 0.0

        self.noise = DynamicParameter.create(value=noise)

        # Action space
        self._init_action_space()

        # Memory
        self.memory = self.get_memory(size=memory_size)

        # Gradient clipping
        self._init_gradient_clipping(clip_norm)

        # Networks (and target-networks)
        self.weights_path = dict(actor=os.path.join(self.base_path, 'actor'),
                                 critic=os.path.join(self.base_path, 'critic'))
        if critic is None:
            critic = dict()
        else:
            assert isinstance(critic, dict)

        if actor is None:
            actor = dict()
        else:
            assert isinstance(actor, dict)

        self.actor = ActorNetwork(agent=self, **actor)
        self.critic = CriticNetwork(agent=self, **critic)

        self.actor_target = ActorNetwork(agent=self, **actor)
        self.critic_target = CriticNetwork(agent=self, **critic)

        # Optimization
        self.optimization_steps = optimization_steps
        self.actor_lr = DynamicParameter.create(value=actor_lr)
        self.critic_lr = DynamicParameter.create(value=critic_lr)

        self.actor_optimizer = utils.get_optimizer_by_name(optimizer, learning_rate=self.actor_lr)
        self.critic_optimizer = utils.get_optimizer_by_name(optimizer, learning_rate=self.critic_lr)
        self.polyak_coeff = polyak

        if load:
            self.load()

    def _init_gradient_clipping(self, clip_norm: Union[float, tuple, None]):
        if clip_norm is None:
            self.should_clip_actor_grads = False
            self.should_clip_critic_grads = False

        elif isinstance(clip_norm, tuple):
            assert len(clip_norm) == 2

            if isinstance(clip_norm[0], float):
                assert clip_norm[0] > 0.0

                self.should_clip_actor_grads = True
                self.actor_clip_norm = tf.constant(clip_norm[0], dtype=tf.float32)
            else:
                self.should_clip_actor_grads = False

            if isinstance(clip_norm[1], float):
                assert clip_norm[1] > 0.0

                self.should_clip_critic_grads = True
                self.critic_clip_norm = tf.constant(clip_norm[1], dtype=tf.float32)
            else:
                self.should_clip_critic_grads = False

        elif isinstance(clip_norm, float):
            assert clip_norm > 0.0

            self.should_clip_actor_grads = True
            self.should_clip_critic_grads = True

            self.actor_clip_norm = tf.constant(clip_norm, dtype=tf.float32)
            self.critic_clip_norm = tf.constant(clip_norm, dtype=tf.float32)
        else:
            raise TypeError(f'`clip_norm` should be "None", "float", or "tuple" not "{type(clip_norm)}"!')

    def _init_action_space(self):
        action_space = self.env.action_space

        if isinstance(action_space, gym.spaces.Box):
            self.num_actions = action_space.shape[0]
            self.action_shape = action_space.shape

            # continuous:
            if action_space.is_bounded():
                self.action_space_type = 'bounded_continuous'

                self.action_low = tf.constant(action_space.low, dtype=tf.float32)
                self.action_high = tf.constant(action_space.high, dtype=tf.float32)
                self.action_range = tf.constant(action_space.high - action_space.low,
                                                dtype=tf.float32)
                # self.convert_action = lambda a: (a * self.action_range + self.action_low)[0].numpy()

                def convert_action(a):
                    # suppose actions a to be in range [0,1]
                    # a = tf.clip_by_value(a, self.action_low, self.action_high)
                    # return (a * self.action_range + self.action_low)[0].numpy()

                    # suppose `actions a` came from `tanh` output
                    a = tf.clip_by_value(a, -1.0, 1.0)
                    return ((a + 1.0) * (self.action_range / 2) + self.action_low)[0].numpy()

                self.convert_action = convert_action
            else:
                self.action_space_type = 'continuous'
                self.convert_action = lambda a: a[0].numpy()
        else:
            # discrete:
            self.action_space_type = 'discrete'

            if isinstance(action_space, gym.spaces.MultiDiscrete):
                # make sure all discrete components of the space have the same number of classes
                assert np.all(action_space.nvec == action_space.nvec[0])

                self.num_actions = action_space.nvec.shape[0]
                self.num_classes = action_space.nvec[0]
                # self.convert_action = lambda a: tf.cast(a[0], dtype=tf.int32).numpy()
                self.action_shape = (self.num_actions, self.num_classes)
            else:
                self.num_actions = 1
                self.num_classes = action_space.n - 1 + 1
                # self.convert_action = lambda a: tf.cast(tf.squeeze(a), dtype=tf.int32).numpy()
                self.action_shape = (self.num_classes,)

            def convert_action(a):
                a = tf.argmax(a, axis=1, output_type=tf.int32)
                return tf.squeeze(a).numpy()

            # def convert_action(a):
            #     a = tf.clip_by_value(a, 0.0, self.num_classes)
            #     a = tf.round(a)
            #     return tf.cast(tf.squeeze(a), dtype=tf.int32).numpy()

            self.convert_action = convert_action

    def get_action(self, states):
        actions = self.actor.actions(inputs=states)
        noise = tf.random.normal(shape=actions.shape, stddev=self.noise())
        return actions + noise

    def get_memory(self, size: int):
        return ReplayMemory(state_spec=self.state_spec, num_actions=self.num_actions, size=size)

    def update(self):
        if len(self.memory) < self.batch_size:
            print('Not updated: few experience in memory!')
            return

        t0 = time.time()

        for _ in range(self.optimization_steps):
            batch = self.memory.sample_batch(batch_size=self.batch_size, seed=self.seed)

            # critic update:
            critic_loss, targets, critic_q_values, critic_grads = self.get_critic_gradients(batch)
            applied_critic_grads = self.apply_critic_gradients(gradients=critic_grads)

            # actor update:
            actor_loss, actions, actor_q_values, actor_grads = self.get_actor_gradients(batch)
            applied_actor_grads = self.apply_actor_gradients(gradients=actor_grads)

            self.update_target_networks()

            self.log(loss_critic=critic_loss, loss_actor=actor_loss, lr_critic=self.critic_lr.value,
                     lr_actor=self.actor_lr.value, q_values_critic=critic_q_values, q_values_actor=actor_q_values,
                     targets=targets, gradients_norm_critic=[tf.norm(g) for g in critic_grads],
                     gradients_norm_actor=[tf.norm(g) for g in actor_grads],
                     gradients_applied_norm_critic=[tf.norm(g) for g in applied_critic_grads],
                     gradients_applied_norm_actor=[tf.norm(g) for g in applied_actor_grads])

            for layer in self.actor.net.layers:
                if isinstance(layer, Dense):
                    weights, bias = layer.get_weights()
                    self.log(**{f'weight-actor_{layer.name}': weights,
                                f'bias-actor_{layer.name}': bias})

            for layer in self.critic.net.layers:
                if isinstance(layer, Dense):
                    weights, bias = layer.get_weights()
                    self.log(**{f'weight-critic_{layer.name}': weights,
                                f'bias-critic_{layer.name}': bias})

        print(f'Update took {round(time.time() - t0, 3)}s.')

    def update_target_networks(self):
        """Updates the weights of the target networks by Polyak average"""
        utils.polyak_averaging(model=self.actor_target.net, old_weights=self.actor.get_weights(),
                               alpha=self.polyak_coeff)
        utils.polyak_averaging(model=self.critic_target.net, old_weights=self.critic.get_weights(),
                               alpha=self.polyak_coeff)

    def get_actor_gradients(self, batch):
        with tf.GradientTape() as tape:
            loss, actions, q_values = self.actor_objective(batch)

        gradients = tape.gradient(loss, self.actor.trainable_variables())
        return loss, actions, q_values, gradients

    def get_critic_gradients(self, batch):
        with tf.GradientTape() as tape:
            loss, targets, q_values = self.critic_objective(batch)

        gradients = tape.gradient(loss, self.critic.trainable_variables())
        return loss, targets, q_values, gradients

    def apply_actor_gradients(self, gradients) -> List[tf.Tensor]:
        if self.should_clip_actor_grads:
            gradients = utils.clip_gradients(gradients, norm=self.actor_clip_norm)

        self.actor_optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables()))
        return gradients

    def apply_critic_gradients(self, gradients) -> List[tf.Tensor]:
        if self.should_clip_critic_grads:
            gradients = utils.clip_gradients(gradients, norm=self.critic_clip_norm)

        self.critic_optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables()))
        return gradients

    # @tf.function
    def actor_objective(self, batch):
        states = batch[:1]

        actions = self.actor.actions(states, training=True)
        q_values = self.critic.q_values(inputs=dict(state=states, action=actions), training=True)

        loss = -tf.reduce_mean(q_values)
        return loss, actions, q_values

    # @tf.function
    def critic_objective(self, batch):
        states, actions, rewards, next_states, terminals = batch

        critic_input = dict(state=states, action=actions)
        target_input = dict(state=next_states, action=self.actor_target.actions(next_states))

        targets = rewards + self.gamma * (1.0 - terminals) * self.critic_target.q_values(inputs=target_input)
        q_values = self.critic.q_values(inputs=critic_input, training=True)

        loss = tf.reduce_mean(losses.MSE(q_values, targets))
        return loss, targets, q_values

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

                    self.log(actions=action, action_env=action_env, noise=self.noise.value, rewards=reward)

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

    def load_weights(self):
        print('[DDPG] loading weights...')
        self.actor.load_weights()
        self.critic.load_weights()

        self.actor_target.set_weights(weights=self.actor.get_weights())
        self.critic_target.set_weights(weights=self.critic.get_weights())

    def save_weights(self):
        self.actor.save_weights()
        self.critic.save_weights()

    def summary(self):
        self.actor.summary()
        self.critic.summary()


class ActorNetwork(Network):
    def __init__(self, agent: DDPGAgent, **kwargs):
        super().__init__(agent=agent)

        self.net = self.build(**kwargs)

    @tf.function
    def actions(self, inputs: Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]], training=False):
        return self.net(inputs, training=training)

    def build(self, **kwargs) -> Model:
        inputs = self._get_input_layers()
        last_layer = self.layers(inputs, **kwargs)
        actions = self.action_layer(layer=last_layer)

        return Model(inputs, outputs=actions, name='Actor-Network')

    def action_layer(self, layer: Layer) -> Layer:
        space_type = self.agent.action_space_type
        num_actions = self.agent.num_actions

        if space_type == 'discrete':
            num_classes = tf.cast(self.agent.num_classes, dtype=tf.float32)

            # actions = Dense(units=num_actions * num_classes, bias_initializer='glorot_uniform',
            #                 activation=lambda x: num_classes * tf.nn.sigmoid(x), name='discrete_actions')(layer)

            actions = Dense(units=num_actions * num_classes, bias_initializer='glorot_uniform',
                            activation='softmax', name='action-logits')(layer)

            # actions = tf.cast(tf.argmax(logits, axis=1), dtype=tf.float32)
            # actions.set_shape(shape=(num_actions,))

            # if num_actions > 1:
            #     actions = Reshape((num_actions, num_classes))(actions)

        elif space_type == 'bounded_continuous':
            # `tanh` layer to bound actions in [-1,+1] as in the paper
            actions = Dense(units=num_actions, activation=tf.nn.tanh, name='bounded_actions',
                            bias_initializer='glorot_uniform')(layer)
        else:
            # continuous
            actions = Dense(units=num_actions, activation=None, name='continuous_actions',
                            bias_initializer='glorot_uniform')(layer)
        return actions

    def layers(self, inputs: Dict[str, Input], **kwargs) -> Layer:
        num_layers = kwargs.get('num_layers', kwargs.get('layers', 2))  # 'num_layers' or 'layers'
        dropout_rate = kwargs.get('dropout', 0.0)
        dense_args = dict(units=kwargs.get('units', 64),
                          activation=kwargs.get('activation', tf.nn.relu),
                          kernel_initializer=kwargs.get('kernel_initializer', 'glorot_uniform'),
                          bias_initializer=kwargs.get('bias_initializer', 'glorot_uniform'))

        x = Dense(**dense_args)(inputs['state'])
        x = BatchNormalization()(x)

        for _ in range(num_layers):
            if dropout_rate > 0.0:
                x = Dense(**dense_args)(x)
                x = Dropout(rate=dropout_rate)(x)
            else:
                x = Dense(**dense_args)(x)

            x = BatchNormalization()(x)

        return x

    def trainable_variables(self):
        return self.net.trainable_variables

    def get_weights(self):
        return self.net.get_weights()

    def set_weights(self, weights):
        return self.net.set_weights(weights)

    def load_weights(self):
        self.net.load_weights(filepath=self.agent.weights_path['actor'], by_name=False)

    def save_weights(self):
        self.net.save_weights(filepath=self.agent.weights_path['actor'])

    def summary(self):
        print("==== [DDPG] Actor Network ====")
        self.net.summary()


# TODO: include L2 weight-decay
class CriticNetwork(Network):
    def __init__(self, agent: DDPGAgent, **kwargs):
        super().__init__(agent=agent)
        self.net = self.build(**kwargs)

    @tf.function
    def q_values(self, inputs: Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]], training=False):
        return self.net(inputs, training=training)

    def build(self, **kwargs) -> Model:
        inputs = self._get_input_layers()
        inputs['action'] = self._action_input_layer()

        last_layer = self.layers(inputs, **kwargs)
        q_values = self.q_layer(layer=last_layer)

        return Model(inputs, outputs=q_values, name='Critic-Network')

    def q_layer(self, layer: Layer) -> Layer:
        return Dense(units=1, bias_initializer='glorot_uniform', name='q_values')(layer)

    def layers(self, inputs: Dict[str, Input], **kwargs) -> Layer:
        units = kwargs.get('units', 64)
        units_action = kwargs.get('units_action', 16)
        num_layers = kwargs.get('num_layers', kwargs.get('layers', 2))  # 'num_layers' or 'layers'
        dense_args = dict(activation=kwargs.get('activation', tf.nn.relu),
                          bias_initializer=kwargs.get('bias_initializer', 'glorot_uniform'),
                          kernel_initializer=kwargs.get('kernel_initializer', 'glorot_uniform'))
        dropout_rate = kwargs.get('dropout', 0.0)

        state_branch = self._branch(inputs['state'], units, num_layers, dropout_rate, **dense_args)
        action_branch = self._branch(inputs['action'], units_action, num_layers, dropout_rate, **dense_args)

        x = Dense(units, **dense_args)(concatenate([state_branch, action_branch]))
        x = BatchNormalization()(x)
        return x

    def _branch(self, input_layer: Input, units: int, num_layers: int, dropout_rate: float, **kwargs) -> Layer:
        x = Dense(units, **kwargs)(input_layer)
        x = BatchNormalization()(x)

        for _ in range(num_layers):
            if dropout_rate > 0.0:
                x = Dense(units, **kwargs)(x)
                x = Dropout(rate=dropout_rate)(x)
            else:
                x = Dense(units, **kwargs)(x)

            x = BatchNormalization()(x)

        return x

    # TODO: does not work for complex action-spaces
    def _action_input_layer(self) -> Input:
        return Input(shape=self.agent.action_shape, dtype=tf.float32, name='action')

    def trainable_variables(self):
        return self.net.trainable_variables

    def get_weights(self):
        return self.net.get_weights()

    def set_weights(self, weights):
        return self.net.set_weights(weights)

    def load_weights(self):
        self.net.load_weights(filepath=self.agent.weights_path['critic'], by_name=False)

    def save_weights(self):
        self.net.save_weights(filepath=self.agent.weights_path['critic'])

    def summary(self):
        print("==== [DDPG] Critic Network ====")
        self.net.summary()
