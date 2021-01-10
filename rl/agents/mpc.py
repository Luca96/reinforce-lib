"""Model-based Model Predictive Control (MPC) Agent"""

import os
import gym
import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from typing import Union, List, Dict, Tuple

from rl import utils
from rl.agents import Agent
from rl.parameters import DynamicParameter
from rl.networks.networks import Network

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import losses
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class MPCAgent(Agent):
    def __init__(self, *args, reward_fn, optimizer='adam', name='mpc-agent', load=False, clip_norm: float = None,
                 lr: Union[float, LearningRateSchedule, DynamicParameter] = 0.001, polyak=1.0, optimization_steps=8,
                 planning_horizon=16, plan_trajectories=64, noise=0.005, network: dict = None, repeat_action=1,
                 discount=1.0, **kwargs):
        assert 0.0 < polyak <= 1.0
        assert repeat_action >= 1
        assert optimization_steps >= 1
        assert 0.0 < discount <= 1.0

        super().__init__(*args, name=name, **kwargs)

        self.repeat_action = repeat_action
        self.memory_size = self.batch_size * optimization_steps
        self.drop_batch_remainder = True

        # MPC planning:
        self.horizon = planning_horizon
        self.num_trajectories = plan_trajectories
        self.reward_fn = reward_fn
        self.noise = DynamicParameter.create(value=noise)
        self.discount = discount

        # Action space
        self._init_action_space()

        # Next state space
        self.next_state_spec = utils.space_to_flat_spec2(space=self.env.observation_space, name='state')

        # Memory
        self.memory = self.get_memory()

        # Gradient clipping
        self._init_gradient_clipping(clip_norm)

        # Network
        self.weights_path = dict(dynamics=os.path.join(self.base_path, 'dynamics'))
        self.dynamics = DynamicsNetwork(agent=self, **(dict() if network is None else network))

        # Optimization
        self.lr = DynamicParameter.create(value=lr)
        self.optimizer = utils.get_optimizer_by_name(optimizer, learning_rate=self.lr)
        self.polyak_coeff = polyak

        if load:
            self.load()

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
            self.distribution_type = 'categorical'

            if isinstance(action_space, gym.spaces.MultiDiscrete):
                # make sure all discrete components of the space have the same number of classes
                assert np.all(action_space.nvec == action_space.nvec[0])

                self.num_actions = action_space.nvec.shape[0]
                self.num_classes = action_space.nvec[0] + 1
                self.action_shape = (self.num_actions, self.num_classes - 1)
            else:
                self.num_actions = 1
                self.num_classes = action_space.n
                self.action_shape = (self.num_classes - 1,)

            def convert_action(a):
                a = tf.round(a)
                return tf.cast(tf.squeeze(a), dtype=tf.int32).numpy()

            self.convert_action = convert_action

    def _init_gradient_clipping(self, clip_norm: Union[None, float]):
        if clip_norm is None:
            self.should_clip_gradients = False

        elif isinstance(clip_norm, float):
            assert clip_norm > 0.0

            self.should_clip_gradients = True
            self.clip_norm = tf.constant(clip_norm, dtype=tf.float32)
        else:
            raise TypeError(f'`clip_norm` should be "None" or "float" not "{type(clip_norm)}"!')

    def get_action(self, state):
        return self.mpc_planning(initial_state=state)

    def mpc_planning(self, initial_state):
        """A.k.a. Random shooting method"""
        best_action = None
        best_reward = -2**64 - 1

        for t in range(self.num_trajectories):
            state = initial_state
            initial_action = self.env.action_space.sample()
            action = initial_action
            total_reward = 0.0
            discount = 1.0

            for h in range(self.horizon):
                inputs = dict(state=state,
                              action=tf.reshape(tf.cast(action, dtype=tf.float32), shape=(1, 1)))

                state = self.dynamics.predict(state)
                # state = self.dynamics.predict(inputs)
                total_reward += discount * self.reward_fn(state, action)
                action = self.env.action_space.sample()
                discount *= self.discount

            if total_reward > best_reward:
                best_action = initial_action
                best_reward = total_reward

        return best_action

    def get_memory(self, *args, **kwargs):
        return MPCMemory(state_spec=self.state_spec, num_actions=self.num_actions, size=self.memory_size)

    def update(self):
        t0 = time.time()

        for batch in self.get_batches():
            gradients, debug = self.get_gradients(batch)
            applied_gradients = self.apply_gradients(gradients)

            self.log(applied_gradients_norm=[tf.norm(g) for g in applied_gradients], **debug)

        print(f'Update took {round(time.time() - t0, 3)}s.')

    @tf.function
    def get_gradients(self, batch):
        with tf.GradientTape() as tape:
            loss, debug = self.dynamics_objective(batch)

        gradients = tape.gradient(loss, self.dynamics.trainable_variables())
        debug['gradients_norm'] = [tf.norm(g) for g in gradients]

        return gradients, debug

    def apply_gradients(self, gradients):
        if self.should_clip_gradients:
            gradients = utils.clip_gradients(gradients, norm=self.clip_norm)

        self.optimizer.apply_gradients(zip(gradients, self.dynamics.trainable_variables()))
        return gradients

    def get_batches(self):
        return utils.data_to_batches(tensors=self.get_tensors(), batch_size=self.batch_size,
                                     shuffle_batches=False, seed=self.seed, shuffle=True,
                                     drop_remainder=self.drop_batch_remainder)

    def get_tensors(self):
        states = self.memory.states[:-1]
        actions = self.memory.actions[:-1]
        next_states = self.memory.states[1:]

        return states, actions, next_states

    @tf.function
    def dynamics_objective(self, batch):
        states, actions, next_states = batch

        predicted_states = self.dynamics.predict(states, training=True)
        loss = 0.5 * losses.MSE(y_true=next_states, y_pred=predicted_states)

        debug = dict(loss=loss, states=states, actions=actions,
                     next_states=next_states, predicted_states=predicted_states)

        return loss, debug

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
                print(f'Episode {episode}')
                preprocess_fn = self.preprocess()
                self.reset()

                state = self.env.reset()
                episode_reward = 0.0
                t0 = time.time()
                render = episode % render_every == 0

                for t in range(1, timesteps + 1):
                    print(f'\t- Timestep {t}')
                    if render:
                        self.env.render()

                    if isinstance(state, dict):
                        state = {f'state_{k}': v for k, v in state.items()}

                    # state = preprocess_fn(state)
                    # state = utils.to_tensor(state)

                    state = np.expand_dims(state, axis=0)

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

                    # next_state = preprocess_fn(next_state)
                    # next_state = utils.to_tensor(next_state)

                    self.memory.append(state, action)
                    state = next_state

                    # check whether a termination (terminal state or end of a transition) is reached:
                    if terminal or (t == timesteps):
                        print(f'Episode {episode} terminated after {t} timesteps in {round((time.time() - t0), 3)}s ' +
                              f'with reward {round(episode_reward, 3)}.')

                        if isinstance(state, dict):
                            state = {f'state_{k}': v for k, v in state.items()}

                        state = np.expand_dims(state, axis=0)

                        self.memory.append(state, action)
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
        print('[MPC] loading weights...')
        self.dynamics.load_weights()

    def save_weights(self):
        self.dynamics.save_weights()

    def summary(self):
        self.dynamics.summary()


class DynamicsNetwork(Network):
    def __init__(self, agent: MPCAgent, **kwargs):
        super().__init__(agent=agent)

        self.weights_path = self.agent.weights_path['dynamics']
        self.net = self.build(**kwargs)

    # @tf.function
    # def predict(self, inputs: Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]], training=False):
    #     # TODO: use multiplication instead? (more expressive??)
    #     next_state = inputs['state'] + self.net(inputs, training=training)
    #
    #     return next_state

    # @tf.function
    def predict(self, state, training=False):
        # TODO: use multiplication instead? (more expressive??)
        next_state = state + self.net(state, training=training)['state']

        return next_state

    def build(self, **kwargs) -> Model:
        inputs = self._get_input_layers()
        inputs['action'] = Input(shape=self.agent.action_shape, dtype=tf.float32, name='action')

        last_layer = self.layers(inputs, **kwargs)
        next_state = self.output_layer(last_layer)

        return Model(inputs['state'], outputs=next_state, name='Dynamics-Network')

    def layers(self, inputs: Dict[str, Input], **kwargs) -> Layer:
        units = kwargs.pop('units', 64)
        num_layers = kwargs.pop('num_layers', kwargs.pop('layers', 2))
        activation = kwargs.pop('activation', tf.nn.relu)
        dropout_rate = kwargs.pop('dropout', 0.0)
        bias_initializer = kwargs.pop('bias_initializer', 'glorot_uniform')

        xs = BatchNormalization()(inputs['state'])
        xs = Dense(units, activation='linear', bias_initializer=bias_initializer, **kwargs)(xs)

        # xa = BatchNormalization()(inputs['action'])
        # xa = Dense(units, activation='linear', bias_initializer=bias_initializer, **kwargs)(xa)
        # xa = Dense(units, activation='linear', bias_initializer=bias_initializer, **kwargs)(xa)

        # x = multiply([xs, xa])
        x = xs

        for _ in range(num_layers):
            if dropout_rate > 0.0:
                x = Dense(units, bias_initializer=bias_initializer, activation=activation, **kwargs)(x)
                x = Dropout(rate=dropout_rate)(x)
            else:
                x = Dense(units, bias_initializer=bias_initializer, activation=activation, **kwargs)(x)

            x = BatchNormalization()(x)

        return x

    def output_layer(self, layer: Layer) -> Dict[str, Layer]:
        outputs = dict()

        for name, spec in self.agent.next_state_spec.items():
            shape = spec['shape']
            v = Dense(units=np.prod(shape), activation='linear', bias_initializer='glorot_uniform')(layer)
            v = Reshape(shape)(v)
            outputs[name] = v

        return outputs

    def trainable_variables(self):
        return self.net.trainable_variables

    def load_weights(self):
        self.net.load_weights(filepath=self.weights_path, by_name=False)

    def save_weights(self):
        self.net.save_weights(filepath=self.weights_path)

    def summary(self):
        print('==== [MPC] Dynamics Network ====')
        self.net.summary()


class MPCMemory:
    def __init__(self, state_spec: dict, num_actions: int, size: int):
        assert size > 0
        self.size = size
        self.action_shape = (1, num_actions)

        if list(state_spec.keys()) == ['state']:
            self.simple_state = True
            self.states = tf.zeros(shape=(0,) + state_spec.get('state'), dtype=tf.float32)
        else:
            self.simple_state = False
            self.states = dict()

            for name, shape in state_spec.items():
                self.states[name] = tf.zeros(shape=(0,) + shape, dtype=tf.float32)

        self.actions = tf.zeros(shape=(0, num_actions), dtype=tf.float32)

    def __len__(self):
        return self.actions.shape[0]

    def append(self, state, action):
        action = tf.reshape(tf.cast(action, dtype=tf.float32), shape=self.action_shape)

        if self.simple_state:
            self.states = tf.concat([self.states, state], axis=0)
        else:
            assert isinstance(state, dict)

            for k, v in state.items():
                self.states[k] = tf.concat([self.states[k], v], axis=0)

        self.actions = tf.concat([self.actions, action], axis=0)

    def ensure_space(self):
        elements_to_remove = len(self) - self.size

        if elements_to_remove > 0:
            if self.simple_state:
                self.states = self.states[elements_to_remove:]
            else:
                for k in self.states.keys():
                    self.states[k] = self.states[k][elements_to_remove:]

            self.actions = self.actions[elements_to_remove:]
