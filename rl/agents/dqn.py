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
    # TODO: add `multi-step targets` (see "rainbow" paper)
    # TODO: add `dueling architecture`
    def __init__(self, *args, lr: Union[float, LearningRateSchedule, DynamicParameter] = 1e-3, name='dqn-agent',
                 gamma=0.99, load=False, optimizer='adam', clip_norm=1.0, polyak=0.999, repeat_action=1,
                 network: dict = None, memory_size=1024, initial_random_batches=0,
                 epsilon: Union[float, LearningRateSchedule, DynamicParameter] = 0.05, **kwargs):
        assert 0.0 < polyak <= 1.0
        assert repeat_action >= 1
        assert initial_random_batches >= 0

        super().__init__(*args, name=name, **kwargs)
        assert memory_size > self.batch_size

        self.gamma = gamma
        self.repeat_action = repeat_action
        self.drop_batch_remainder = memory_size % self.batch_size != 0
        self.initial_random_steps = initial_random_batches * self.batch_size

        # Epsilon-greedy probability
        if isinstance(epsilon, float):
            assert 0.0 <= epsilon < 1.0

        self.epsilon = DynamicParameter.create(value=epsilon)

        # Action space
        self._init_action_space()

        # Memory
        self.memory = self.get_memory(size=memory_size)

        # Gradient clipping
        self._init_gradient_clipping(clip_norm)

        # Networks (DQN and target-network)
        self.weights_path = dict(dqn=os.path.join(self.base_path, 'dqn'))
        self.dqn, self.target = self.get_network(**(network if isinstance(network, dict) else {}))

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
        assert isinstance(self.env.action_space, gym.spaces.Discrete)

        self.num_actions = 1
        self.num_classes = self.env.action_space.n
        self.convert_action = lambda a: tf.cast(tf.squeeze(a), dtype=tf.int32).numpy()

    def get_action(self, state):
        if self.epsilon() > utils.tf_chance(seed=self.seed):
            # TODO: Sampling Gym space is not deterministic (lacks `seed` parameter)
            return self.env.action_space.sample()

        return self.dqn.greedy_actions(state, training=False)

    def get_network(self, **kwargs):
        dqn = DQNetwork(agent=self, **kwargs)
        target = DQNetwork(agent=self, **kwargs)
        target.set_weights(weights=dqn.get_weights())

        return dqn, target

    def get_memory(self, size: int):
        return ReplayMemory(state_spec=self.state_spec, num_actions=self.num_actions, size=size)

    def update(self):
        if len(self.memory) < self.batch_size:
            print('Not updated: memory too small!')
            return

        t0 = time.time()

        batch = self.memory.sample_batch(batch_size=self.batch_size, seed=self.seed)
        loss, gradients, debug = self.get_gradients(batch)
        applied_grads, debug = self.apply_gradients(gradients, debug)

        self.update_target_network()
        self.log(average=True, lr=self.lr.value, **debug)

        print(f'Update took {round(time.time() - t0, 3)}s.')

    def update_target_network(self):
        """Updates the weights of the target networks by Polyak average"""
        utils.polyak_averaging2(model=self.dqn, target=self.target, alpha=self.polyak_coeff)

    def get_gradients(self, batch):
        with tf.GradientTape() as tape:
            loss, debug = self.objective(batch)

        gradients = tape.gradient(loss, self.dqn.trainable_variables())
        debug['gradient_norm'] = [tf.norm(g) for g in gradients]

        return loss, gradients, debug

    def apply_gradients(self, gradients, debug):
        if self.should_clip_grads:
            gradients = utils.clip_gradients(gradients, norm=self.grad_norm_value)
            debug['gradient_clipped_norm'] = [tf.norm(g) for g in gradients]

        self.optimizer.apply_gradients(zip(gradients, self.dqn.trainable_variables()))
        return gradients, debug

    @tf.function
    def objective(self, batch):
        """DQN's Training objective"""
        states, actions, rewards, next_states, terminals = batch[:5]

        q_values = self.dqn.q_values(states, training=True)
        selected_q_values = self.index_q_values(q_values, actions)

        targets = self.targets(next_states, rewards, terminals)

        loss = 0.5 * losses.MSE(selected_q_values, targets)
        debug = dict(loss=loss, targets=targets, q_values=selected_q_values)

        return loss, debug

    def index_q_values(self, q_values, actions):
        """Use `actions` to index a tensor of Q-values"""
        shape = (q_values.shape[0], 1)

        q_indices = tf.concat([
            tf.reshape(tf.range(start=0, limit=shape[0], dtype=tf.int32), shape),
            tf.cast(actions, dtype=tf.int32)
        ], axis=1)

        return tf.gather_nd(q_values, q_indices)

    @tf.function
    def targets(self, next_states, rewards, terminals):
        q_values = self.target.q_values(next_states, training=False)

        targets = rewards + self.gamma * tf.reduce_max(q_values, axis=1, keepdims=True)
        targets = tf.where(terminals == 0.0, x=rewards, y=targets)

        return tf.stop_gradient(targets)

    # TODO: check
    def random_steps(self):
        preprocess_fn = self.preprocess()
        self.reset()

        state = self.env.reset()
        state = preprocess_fn(state)
        state = utils.to_tensor(state)

        episode_reward = 0.0
        t0 = time.time()
        t = 0

        for _ in range(self.initial_random_steps):
            action = self.env.action_space.sample()
            self.log(random_action=action)

            for _ in range(self.repeat_action):
                next_state, reward, terminal, _ = self.env.step(action)
                episode_reward += reward

                if terminal:
                    break

            next_state = preprocess_fn(next_state)
            next_state = utils.to_tensor(next_state)

            self.memory.append(state, action, reward, next_state, terminal)
            state = next_state

            if terminal:
                print(f'Random episode terminated after {t} timesteps in {round((time.time() - t0), 3)}s ' +
                      f'with reward {round(episode_reward, 3)}.')

                self.log(random_reward=episode_reward)
                self.write_summaries()

                self.memory.ensure_space()
                self.on_episode_end()

                preprocess_fn = self.preprocess()
                self.reset()
                state = self.env.reset()
                state = preprocess_fn(state)
                state = utils.to_tensor(state)

                t0 = time.time()
                t = 0
                episode_reward = 0.0
            else:
                t += 1

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
            if self.initial_random_steps > 0:
                self.random_steps()

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
    # TODO: generic save/load key, summary name
    def __init__(self, agent: DQNAgent, **kwargs):
        super().__init__(agent)
        self.agent: DQNAgent

        # Deep Q-Network
        self.net = self.build(**kwargs)

    @tf.function
    def q_values(self, inputs: Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]], training=False):
        return self.net(inputs, training=training)

    @tf.function
    def greedy_actions(self, inputs: Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]], training=False):
        q = self.q_values(inputs, training=training)
        return tf.argmax(q, axis=1)

    # TODO: only 1-D discrete actions supported
    def build(self, **kwargs) -> Model:
        assert self.agent.num_actions == 1
        q_args = kwargs.pop('output', dict(bias_initializer='zeros', kernel_initializer='glorot_uniform'))

        inputs = self._get_input_layers()
        last_layer = self.layers(inputs, **kwargs)
        q_values = self.output_layer(last_layer, **q_args)

        return Model(inputs, outputs=q_values, name='Deep-Q-Network')

    # def layers(self, inputs: Dict[str, Input], **kwargs) -> Layer:
    #     units = kwargs.get('units', 64)
    #     num_layers = kwargs.get('num_layers', kwargs.get('layers', 2))  # 'num_layers' or 'layers'
    #     dropout_rate = kwargs.get('dropout', 0.0)
    #     normalization = kwargs.get('normalization', 'layer')
    #     args = dict(activation=kwargs.get('activation', tf.nn.tanh),
    #                 bias_initializer=kwargs.get('bias_initializer', 'glorot_uniform'),
    #                 kernel_initializer=kwargs.get('kernel_initializer', 'glorot_normal'))
    #
    #     x = Dense(units, **args)(inputs['state'])
    #     x = utils.normalization_layer(x, name=normalization)
    #
    #     for _ in range(num_layers):
    #         if dropout_rate > 0.0:
    #             x = Dense(units, **args)(x)
    #             x = Dropout(rate=dropout_rate)(x)
    #         else:
    #             x = Dense(units, **args)(x)
    #
    #         x = utils.normalization_layer(x, name=normalization)
    #
    #     return x

    def layers(self, inputs, **kwargs):
        from rl.layers import NoisyDense

        num_layers = kwargs.get('num_layers', kwargs.get('layers', 2))  # 'num_layers' or 'layers'
        normalization = kwargs.get('normalization', 'layer')
        args = dict(activation=kwargs.get('activation', 'relu'), noise='independent',
                    units=kwargs.get('units', 64))

        x = NoisyDense(**args)(inputs['state'])
        x = utils.normalization_layer(x, name=normalization)

        for _ in range(num_layers):
            x = NoisyDense(**args)(x)
            x = utils.normalization_layer(x, name=normalization)

        return x

    def output_layer(self, layer: Layer, name='q_values', **kwargs) -> Layer:
        return Dense(units=self.agent.num_classes, name=name, **kwargs)(layer)

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
        self.action_shape = (1, num_actions)

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
        action = tf.reshape(tf.identity(action), shape=self.action_shape)
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
