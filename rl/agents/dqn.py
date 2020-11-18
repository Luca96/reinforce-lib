"""Deep Q-Learning (DQN) with Experience Replay"""

import os
import gym
import time
import numpy as np
import tensorflow as tf

from typing import Union, List, Dict

from rl import utils
from rl.agents import Agent
from rl.parameters import DynamicParameter
from rl.networks.networks import Network

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import losses
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class DQNAgent(Agent):
    # TODO: fix retracing issue (functions: objective, targets, and q_values)
    # TODO: support for continuous actions?
    def __init__(self, *args, lr: Union[float, LearningRateSchedule, DynamicParameter] = 1e-3, name='dqn-agent',
                 gamma=0.99, load=False, optimizer='adam', clip_norm=1.0, polyak=0.999, repeat_action=1,
                 memory_size=1000, epsilon: Union[float, LearningRateSchedule, DynamicParameter] = 0.05, **kwargs):
        assert 0.0 < polyak <= 1.0
        assert repeat_action >= 1

        super().__init__(*args, name=name, **kwargs)
        assert memory_size > self.batch_size

        self.gamma = gamma
        self.repeat_action = repeat_action
        self.drop_batch_remainder = memory_size % self.batch_size != 0

        # Epsilon-greedy probability
        if isinstance(epsilon, float):
            assert 0.0 <= epsilon < 1.0

        self.epsilon = DynamicParameter.create(value=epsilon)

        # Action space
        self._init_action_space()

        # Memory
        self.memory = ReplayMemory(state_spec=self.state_spec, num_actions=self.num_actions, size=memory_size)

        # Gradient clipping
        self._init_gradient_clipping(clip_norm)

        # Networks (DQN and target-network)
        self.weights_path = dict(dqn=os.path.join(self.base_path, 'dqn'))
        self.dqn = DQNetwork(agent=self)
        self.target = DQNetwork(agent=self)

        # Optimization
        self.lr = DynamicParameter.create(value=lr)
        self.optimizer = utils.get_optimizer_by_name(optimizer, learning_rate=self.lr)
        self.polyak_coeff = polyak

        if load:
            self.load()

    def _init_gradient_clipping(self, clip_norm: Union[float, None]):
        if clip_norm is None:
            self.should_clip_grads = False

        elif isinstance(clip_norm, float):
            assert clip_norm > 0.0

            self.should_clip_grads = True
            self.grad_norm_value = tf.constant(clip_norm, dtype=tf.float32)
        else:
            raise TypeError(f'`clip_norm` should be "None" or "float" not "{type(clip_norm)}"!')

    def _init_action_space(self):
        action_space = self.env.action_space

        if isinstance(action_space, gym.spaces.Box):
            raise NotImplementedError('`gym.spaces.Box` not yet supported!')
            self.num_actions = action_space.shape[0]

            # continuous:
            if action_space.is_bounded():
                self.distribution_type = 'beta'

                self.action_low = tf.constant(action_space.low, dtype=tf.float32)
                self.action_high = tf.constant(action_space.high, dtype=tf.float32)
                self.action_range = tf.constant(action_space.high - action_space.low,
                                                dtype=tf.float32)

                self.convert_action = lambda a: (a * self.action_range + self.action_low)[0].numpy()
            else:
                self.distribution_type = 'gaussian'
                self.convert_action = lambda a: a[0].numpy()
        else:
            # discrete:
            self.distribution_type = 'categorical'

            if isinstance(action_space, gym.spaces.MultiDiscrete):
                raise NotImplementedError('`gym.spaces.MultiDiscrete` not yet supported!')
                # make sure all discrete components of the space have the same number of classes
                assert np.all(action_space.nvec == action_space.nvec[0])

                self.num_actions = action_space.nvec.shape[0]
                self.num_classes = action_space.nvec[0] + 1  # to include the last class, i.e. 0 to K (not 0 to k-1)
                self.convert_action = lambda a: tf.cast(a[0], dtype=tf.int32).numpy()
            else:
                self.num_actions = 1
                self.num_classes = action_space.n
                self.convert_action = lambda a: tf.cast(tf.squeeze(a), dtype=tf.int32).numpy()

    def get_action(self, state):
        if self.epsilon() > utils.tf_chance(seed=self.seed):
            return self.env.action_space.sample()

        q_values = self.dqn.q_values(state, training=False)
        return tf.argmax(q_values, axis=1)

    def update(self):
        if len(self.memory) < self.batch_size:
            print('Not updated: memory too small!')
            return

        t0 = time.time()

        batch = self.memory.sample_batch(batch_size=self.batch_size, seed=self.seed)
        loss, q_values, targets, gradients = self.get_gradients(batch)
        applied_grads = self.apply_gradients(gradients)
        self.update_target_network()

        self.log(loss=loss, q_values=q_values, targets=targets, lr=self.lr.value,
                 gradients_norm=[tf.norm(gradient) for gradient in gradients],
                 gradients_applied_norm=[tf.norm(g) for g in applied_grads])

        print(f'Update took {round(time.time() - t0, 3)}s.')

    def update_target_network(self):
        """Updates the weights of the target network by Polyak average"""
        utils.polyak_averaging(model=self.target.net, old_weights=self.dqn.get_weights(), alpha=self.polyak_coeff)

    def get_gradients(self, batch):
        with tf.GradientTape() as tape:
            loss, q_values, targets = self.objective(batch)

        gradients = tape.gradient(loss, self.dqn.trainable_variables())
        return loss, q_values, targets, gradients

    def apply_gradients(self, gradients):
        if self.should_clip_grads:
            gradients = utils.clip_gradients(gradients, norm=self.grad_norm_value)

        self.optimizer.apply_gradients(zip(gradients, self.dqn.trainable_variables()))
        return gradients

    @tf.function
    def objective(self, batch):
        """DQN's Training objective"""
        states, actions, rewards, next_states, terminals = batch[:5]

        q_values = self.dqn.q_values(states, training=True)

        # use `actions` to index (select) q-values
        shape = (q_values.shape[0], 1)

        q_indices = tf.concat([
            tf.reshape(tf.range(start=0, limit=shape[0], dtype=tf.int32), shape),
            tf.cast(actions, dtype=tf.int32)
        ], axis=1)

        selected_q_values = tf.gather_nd(q_values, q_indices)
        targets = self.targets(next_states, rewards, terminals)

        loss = 0.5 * losses.MSE(selected_q_values, targets)
        return loss, selected_q_values, targets

    @tf.function
    def targets(self, next_states, rewards, terminals):
        q_values = self.target.q_values(next_states, training=True)
        targets = rewards + self.gamma * tf.reduce_max(q_values, axis=1, keepdims=True)
        targets = tf.where(terminals == 0.0, x=rewards, y=targets)
        return targets

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

                    self.log(actions=action, action_env=action_env, epsilon=self.epsilon.value, rewards=reward)

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
        print('[DQN] loading weights...')
        self.dqn.load_weights()
        self.target.set_weights(weights=self.dqn.get_weights())

    def save_weights(self):
        self.dqn.save_weights()

    def summary(self):
        self.dqn.summary()


class DQNetwork(Network):
    def __init__(self, agent: DQNAgent):
        super().__init__(agent)
        self.agent: DQNAgent

        self.distribution = self.agent.distribution_type

        # Deep Q-Network
        self.net = self.build()

    @tf.function
    def q_values(self, inputs: Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]], training=False):
        return self.net(inputs, training=training)

    # TODO: only 1-D discrete actions supported
    def build(self, **kwargs) -> Model:
        assert self.distribution == 'categorical'
        assert self.agent.num_actions == 1

        inputs = self._get_input_layers()
        last_layer = self.layers(inputs, **kwargs)
        q_values = Dense(units=self.agent.num_classes, activation=None, name='q_values')(last_layer)

        return Model(inputs, outputs=q_values, name='Deep-Q-Network')

    def layers(self, inputs: Dict[str, Input], **kwargs) -> Layer:
        """Defines the architecture of the DQN"""
        units = kwargs.get('units', 32)
        num_layers = kwargs.get('num_layers', kwargs.get('layers', 2))  # 'num_layers' or 'layers'
        activation = kwargs.get('activation', tf.nn.swish)
        dropout_rate = kwargs.get('dropout', 0.0)
        linear_units = kwargs.get('linear_units', 0)

        # inputs = concatenate([inputs['state'], inputs['action']], axis=1)
        # x = Dense(units, activation=activation)(inputs)
        x = Dense(units, activation=activation)(inputs['state'])
        x = BatchNormalization()(x)

        for _ in range(0, num_layers, 2):
            if dropout_rate > 0.0:
                x = Dense(units, activation=activation)(x)
                x = Dropout(rate=dropout_rate)(x)

                x = Dense(units, activation=activation)(x)
                x = Dropout(rate=dropout_rate)(x)
            else:
                x = Dense(units, activation=activation)(x)
                x = Dense(units, activation=activation)(x)

            x = BatchNormalization()(x)

        if linear_units > 0:
            x = Dense(units=linear_units, activation='linear')(x)

        return x

    def trainable_variables(self):
        return self.net.trainable_variables

    def get_weights(self):
        return self.net.get_weights()

    def set_weights(self, weights):
        return self.net.set_weights(weights)

    def load_weights(self):
        self.net.load_weights(filepath=self.agent.weights_path['dqn'], by_name=False)

    def save_weights(self):
        self.net.save_weights(filepath=self.agent.weights_path['dqn'])

    def summary(self):
        print("==== Deep Q-Network ====")
        self.net.summary()


class ReplayMemory:
    def __init__(self, state_spec: dict, num_actions: int, size: int):
        assert size > 0
        self.size = size

        if list(state_spec.keys()) == ['state']:
            # Simple state-space
            self.states = tf.zeros(shape=(0,) + state_spec.get('state'), dtype=tf.float32)
            self.next_states = tf.zeros_like(self.states)
            self.simple_state = True
        else:
            # Complex state-space
            self.states = dict()
            self.next_states = dict()
            self.simple_state = False

            for name, shape in state_spec.items():
                self.states[name] = tf.zeros(shape=(0,) + shape, dtype=tf.float32)
                self.next_states[name] = tf.zeros_like(self.states[name])

        self.actions = tf.zeros(shape=(0, num_actions), dtype=tf.float32)
        self.rewards = tf.zeros(shape=(0, 1), dtype=tf.float32)
        self.terminals = tf.zeros(shape=(0, 1), dtype=tf.float32)

    def __len__(self):
        """Memory's current size"""
        return self.actions.shape[0]

    def append(self, state, action, reward, next_state, terminal):
        action = tf.reshape(action, shape=(1, self.actions.shape[1]))
        reward = tf.reshape(reward, shape=(1, 1))
        terminal = tf.reshape(terminal, shape=(1, 1))

        if self.simple_state:
            self.states = tf.concat([self.states, state], axis=0)
            self.next_states = tf.concat([self.next_states, next_state], axis=0)
        else:
            assert isinstance(state, dict) and isinstance(next_state, dict)

            for k, v in state.items():
                self.states[k] = tf.concat([self.states[k], v], axis=0)
                self.next_states[k] = tf.concat([self.next_states[k], next_state[k]], axis=0)

        self.actions = tf.concat([self.actions, tf.cast(action, dtype=tf.float32)], axis=0)
        self.rewards = tf.concat([self.rewards, tf.cast(reward, dtype=tf.float32)], axis=0)
        self.terminals = tf.concat([self.terminals, tf.cast(terminal, dtype=tf.float32)], axis=0)

    # TODO: in this way `obs_skipping` is not possible, anymore...
    def sample_batch(self, batch_size: int, seed=None) -> tuple:
        assert len(self) >= batch_size

        # use random indices to randomly sample (get) items
        indices = tf.range(start=0, limit=len(self), dtype=tf.int32)
        indices = tf.random.shuffle(indices, seed=seed)[:batch_size]

        # batch = (states, actions, rewards, next_states, terminals)
        if self.simple_state:
            batch = (tf.gather(self.states, indices),
                     tf.gather(self.actions, indices),
                     tf.gather(self.rewards, indices),
                     tf.gather(self.next_states, indices),
                     tf.gather(self.terminals, indices))
        else:
            batch = ({k: tf.gather(v, indices) for k, v in self.states.items()},
                     tf.gather(self.actions, indices),
                     tf.gather(self.rewards, indices),
                     {k: tf.gather(v, indices) for k, v in self.next_states.items()},
                     tf.gather(self.terminals, indices))
        return batch

    def ensure_space(self):
        elements_to_remove = self.actions.shape[0] - self.size

        # make space for next's episode items, if memory is full
        if elements_to_remove > 0:
            if self.simple_state:
                self.states = self.states[elements_to_remove:]
                self.next_states = self.next_states[elements_to_remove:]
            else:
                for k in self.states.keys():
                    self.states[k] = self.states[k][elements_to_remove:]
                    self.next_states[k] = self.next_states[k][elements_to_remove:]

            self.actions = self.actions[elements_to_remove:]
            self.rewards = self.rewards[elements_to_remove:]
            self.terminals = self.terminals[elements_to_remove:]
