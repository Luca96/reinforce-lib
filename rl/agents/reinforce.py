import os
import gym
import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from typing import Union, List, Dict

from rl import utils
from rl.agents import Agent
from rl.networks import Network
from rl.parameters import DynamicParameter, ConstantParameter, ParameterWrapper

from tensorflow.keras import losses
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class ReinforceAgent(Agent):
    def __init__(self, *args, policy_lr: Union[float, LearningRateSchedule, DynamicParameter] = 1e-3, gamma=0.99,
                 lambda_=0.95, value_lr: Union[float, LearningRateSchedule, DynamicParameter] = 3e-4, load=False,
                 optimization_steps=(1, 1), name='reinforce-agent', optimizer='adam', clip_norm=(1.0, 1.0),
                 entropy_regularization: Union[float, LearningRateSchedule, DynamicParameter] = 0.0, polyak=1.0,
                 network: Union[dict, Network] = None, advantage_scale: Union[float, DynamicParameter] = 2.0,
                 repeat_action=1, episodes_per_update=1, weight_decay=1e-2, **kwargs):
        assert 0.0 < gamma <= 1.0
        assert 0.0 < lambda_ <= 1.0
        assert 0.0 < polyak <= 1.0
        assert repeat_action >= 1
        assert episodes_per_update >= 1
        assert weight_decay >= 0.0

        super().__init__(*args, name=name, **kwargs)

        self.memory: ReinforceMemory = None
        self.gamma = gamma
        self.lambda_ = lambda_
        self.repeat_action = repeat_action
        self.episodes_per_update = episodes_per_update
        self.weight_decay = weight_decay

        if isinstance(advantage_scale, float):
            self.adv_scale = ConstantParameter(value=advantage_scale)
        else:
            self.adv_scale = advantage_scale

        # Entropy regularization
        if isinstance(entropy_regularization, float):
            self.entropy_strength = ConstantParameter(value=entropy_regularization)

        elif isinstance(entropy_regularization, LearningRateSchedule):
            self.entropy_strength = ParameterWrapper(entropy_regularization)

        # Action space
        self._init_action_space()

        print('state_spec:', self.state_spec)
        print('action_shape:', self.num_actions)
        print('distribution:', self.distribution_type)

        # Gradient clipping:
        self._init_gradient_clipping(clip_norm)

        # Networks & Loading
        self.weights_path = dict(policy=os.path.join(self.base_path, 'policy_net'),
                                 value=os.path.join(self.base_path, 'value_net'))

        if isinstance(network, dict):
            network_class = network.pop('network', ReinforceNetwork)

            if network_class is ReinforceNetwork:
                # policy/value-specific arguments
                policy_args = network.pop('policy', {})
                value_args = network.pop('value', policy_args)

                # common arguments
                for k, v in network.items():
                    if k not in policy_args:
                        policy_args[k] = v

                    if k not in value_args:
                        value_args[k] = v

                self.network = network_class(agent=self, policy=policy_args, value=value_args, **network)
            else:
                self.network = network_class(agent=self, **network)
        else:
            self.network = ReinforceNetwork(agent=self, policy={}, value={})

        # Optimization
        self.policy_lr = self._init_lr_schedule(policy_lr)
        self.value_lr = self._init_lr_schedule(value_lr)

        self.policy_optimizer = utils.get_optimizer_by_name(optimizer, learning_rate=self.policy_lr)
        self.value_optimizer = utils.get_optimizer_by_name(optimizer, learning_rate=self.value_lr)
        self.optimization_steps = dict(policy=optimization_steps[0], value=optimization_steps[1])

        self.should_polyak_average = polyak < 1.0
        self.polyak_coeff = polyak

        if load:
            self.load()

    @staticmethod
    def _init_lr_schedule(lr: Union[float, LearningRateSchedule]):
        if isinstance(lr, float):
            return ConstantParameter(lr)

        if isinstance(lr, ParameterWrapper) or isinstance(lr, DynamicParameter):
            return lr

        return ParameterWrapper(schedule=lr)

    def _init_gradient_clipping(self, clip_norm: Union[tuple, float, None]):
        if clip_norm is None:
            self.should_clip_policy_grads = False
            self.should_clip_value_grads = False

        elif isinstance(clip_norm, float):
            assert clip_norm > 0.0
            self.should_clip_policy_grads = True
            self.should_clip_value_grads = True

            self.grad_norm_policy = clip_norm
            self.grad_norm_value = clip_norm
        else:
            assert isinstance(clip_norm, tuple)

            if clip_norm[0] is None:
                self.should_clip_policy_grads = False
            else:
                assert isinstance(clip_norm[0], float)
                assert clip_norm[0] > 0.0

                self.should_clip_policy_grads = True
                self.grad_norm_policy = tf.constant(clip_norm[0], dtype=tf.float32)

            if clip_norm[1] is None:
                self.should_clip_value_grads = False
            else:
                assert isinstance(clip_norm[1], float)
                assert clip_norm[1] > 0.0

                self.should_clip_value_grads = True
                self.grad_norm_value = tf.constant(clip_norm[1], dtype=tf.float32)

    def _init_action_space(self):
        action_space = self.env.action_space

        if isinstance(action_space, gym.spaces.Box):
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
                # make sure all discrete components of the space have the same number of classes
                assert np.all(action_space.nvec == action_space.nvec[0])

                self.num_actions = action_space.nvec.shape[0]
                self.num_classes = action_space.nvec[0] + 1  # to include the last class, i.e. 0 to K (not 0 to k-1)
                self.convert_action = lambda a: tf.cast(a[0], dtype=tf.int32).numpy()
            else:
                self.num_actions = 1
                self.num_classes = action_space.n
                self.convert_action = lambda a: tf.cast(tf.squeeze(a), dtype=tf.int32).numpy()

    def act(self, state, *args, **kwargs):
        return self.network.act(state)

    def predict(self, state, *args, **kwargs):
        return self.network.predict(inputs=state, actions=args[0])

    def update(self):
        t0 = time.time()

        # # Compute advantages and returns:
        # returns = self.memory.compute_returns(discount=self.gamma)
        # values, advantages = self.memory.compute_advantages(self.gamma, self.lambda_, scale=self.adv_scale())
        #
        # self.log(returns=returns, advantage_scale=self.adv_scale.value, advantages_normalized=self.memory.advantages,
        #          returns_base=self.memory.returns[:, 0], returns_exp=self.memory.returns[:, 1], values=values,
        #          values_base=self.memory.values[:, 0], values_exp=self.memory.values[:, 1], advantages=advantages)

        # Prepare data:
        value_batches = self.get_value_batches()
        policy_batches = self.get_policy_batches()

        # Policy network optimization:
        for opt_step in range(self.optimization_steps['policy']):
            for data_batch in policy_batches:
                total_loss, policy_grads = self.get_policy_gradients(data_batch)

                self.apply_policy_gradients(policy_grads)

                self.log(loss_total=total_loss, lr_policy=self.policy_lr.value,
                         gradients_norm_policy=[tf.norm(gradient) for gradient in policy_grads])

        # Value network optimization:
        for _ in range(self.optimization_steps['value']):
            for data_batch in value_batches:
                value_loss, value_grads = self.get_value_gradients(data_batch)

                self.apply_value_gradients(value_grads)

                self.log(loss_total_value=value_loss, lr_value=self.value_lr.value,
                         gradients_norm_value=[tf.norm(gradient) for gradient in value_grads])

        self.network.log_weights()
        print(f'Update took {round(time.time() - t0, 3)}s')

    def get_value_gradients(self, batch):
        with tf.GradientTape() as tape:
            loss = self.value_objective(batch)

        gradients = tape.gradient(loss, self.network.value.trainable_variables)
        return loss, gradients

    def get_policy_gradients(self, batch):
        with tf.GradientTape() as tape:
            loss = self.policy_objective(batch)

        gradients = tape.gradient(loss, self.network.policy.trainable_variables)
        return loss, gradients

    def apply_value_gradients(self, gradients):
        if self.should_clip_value_grads:
            gradients = utils.clip_gradients(gradients, norm=self.grad_norm_value)

        if self.should_polyak_average:
            old_weights = self.network.value.get_weights()
            self.value_optimizer.apply_gradients(zip(gradients, self.network.value.trainable_variables))
            utils.polyak_averaging(self.network.value, old_weights, alpha=self.polyak_coeff)
        else:
            self.value_optimizer.apply_gradients(zip(gradients, self.network.value.trainable_variables))

        return gradients

    def apply_policy_gradients(self, gradients):
        if self.should_clip_policy_grads:
            gradients = utils.clip_gradients(gradients, norm=self.grad_norm_policy)

        if self.should_polyak_average:
            old_weights = self.network.policy.get_weights()
            self.policy_optimizer.apply_gradients(zip(gradients, self.network.policy.trainable_variables))
            utils.polyak_averaging(self.network.policy, old_weights, alpha=self.polyak_coeff)
        else:
            self.policy_optimizer.apply_gradients(zip(gradients, self.network.policy.trainable_variables))

        return gradients

    def value_batch_tensors(self) -> Union[tuple, dict]:
        """Defines which data to use in `get_value_batches()`"""
        return self.memory.states, self.memory.returns

    def policy_batch_tensors(self) -> Union[tuple, dict]:
        """Defines which data to use in `get_policy_batches()`"""
        return self.memory.states, self.memory.actions, self.memory.advantages

    def get_value_batches(self):
        """Computes batches of data for updating the value network"""
        return utils.data_to_batches(tensors=self.value_batch_tensors(), batch_size=self.batch_size,
                                     drop_remainder=self.drop_batch_remainder, skip=self.skip_count,
                                     shuffle=True, shuffle_batches=False, num_shards=self.obs_skipping)

    def get_policy_batches(self):
        """Computes batches of data for updating the policy network"""
        return utils.data_to_batches(tensors=self.policy_batch_tensors(), batch_size=self.batch_size,
                                     drop_remainder=self.drop_batch_remainder, skip=self.skip_count,
                                     num_shards=self.obs_skipping, shuffle=self.shuffle,
                                     shuffle_batches=self.shuffle_batches)
    
    @tf.function
    def _value_objective(self, batch):
        states, returns = batch[:2]
        values = self.network.value_predict(states)

        return 0.5 * tf.reduce_mean(losses.mean_squared_error(y_true=returns, y_pred=values))

    def value_objective(self, batch):
        value_loss = self._value_objective(batch)
        regularization_loss = tf.reduce_sum(self.network.value.losses)

        self.log(loss_value=value_loss, loss_regularization_value=regularization_loss)
        return value_loss + regularization_loss

    def policy_objective(self, batch):
        """Policy gradient objective"""
        states, actions, advantages = batch[:3]
        log_prob, entropy = self.predict(states, actions)

        # entropy
        entropy = tf.reduce_mean(entropy)
        entropy_penalty = self.entropy_strength() * entropy

        # policy gradient
        policy_loss = -tf.reduce_mean(log_prob * advantages)
        regularization_loss = tf.reduce_sum(self.network.policy.losses)
        total_loss = policy_loss - entropy_penalty + regularization_loss

        # Log stuff
        self.log(log_prob=tf.reduce_mean(log_prob), entropy=entropy, entropy_coeff=self.entropy_strength.value,
                 loss_policy=policy_loss.numpy(), loss_entropy=entropy_penalty.numpy(),
                 loss_regularization_policy=regularization_loss)

        return total_loss

    def learn(self, episodes: int, timesteps: int, save_every: Union[bool, str, int] = False,
              render_every: Union[bool, str, int] = False, close=True):
        if save_every is False:
            save_every = episodes + 1
        elif save_every is True:
            save_every = 1
        elif save_every == 'end':
            save_every = episodes

        if render_every is False:
            render_every = episodes + 1
        elif render_every is True:
            render_every = 1

        try:
            for episode in range(1, episodes * self.episodes_per_update + 1):
                preprocess_fn = self.preprocess()
                self.reset()
                # self.memory = ReinforceMemory(state_spec=self.state_spec, num_actions=self.num_actions)
                self.memory = ReinforceMemory(agent=self)

                state = self.env.reset()
                episode_reward = 0.0
                t0 = time.time()
                render = episode % render_every == 0

                for t in range(1, timesteps + 1):
                    if render:
                        self.env.render()

                    if isinstance(state, dict):
                        state = {f'state_{k}': v for k, v in state.items()}

                    state = preprocess_fn(state)
                    state = utils.to_tensor(state)

                    # Agent prediction
                    action, mean, std, value = self.act(state)
                    action_env = self.convert_action(action)

                    # Environment step
                    for _ in range(self.repeat_action):
                        next_state, reward, done, _ = self.env.step(action_env)
                        episode_reward += reward

                        if done:
                            break

                    self.log(actions=action, action_env=action_env, rewards=reward,
                             distribution_mean=mean, distribution_std=std)

                    self.memory.append(state, action, reward, value)
                    state = next_state

                    # check whether a termination (terminal state or end of a transition) is reached:
                    if done or (t == timesteps):
                        print(f'Episode {episode} terminated after {t} timesteps in {round((time.time() - t0), 3)}s ' +
                              f'with reward {round(episode_reward, 3)}.')

                        if isinstance(state, dict):
                            state = {f'state_{k}': v for k, v in state.items()}

                        state = preprocess_fn(state)
                        state = utils.to_tensor(state)

                        last_value = self.network.predict_last_value(state, is_terminal=done)
                        returns, values, advantages = self.memory.end_trajectory(last_value)

                        self.log(returns=returns, advantage_scale=self.adv_scale.value, advantages=advantages,
                                 advantages_normalized=self.memory.advantages, values=values,
                                 returns_base=self.memory.returns[:, 0], returns_exp=self.memory.returns[:, 1],
                                 values_base=self.memory.values[:, 0], values_exp=self.memory.values[:, 1])
                        break

                if episode % self.episodes_per_update == 0:
                    self.update()

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

    def summary(self):
        self.network.summary()

    def load_weights(self):
        self.network.load_weights()

    def save_weights(self):
        self.network.save_weights()

    def save_config(self):
        print('save config')
        self.update_config(policy_lr=self.policy_lr.serialize(), value_lr=self.value_lr.serialize(),
                           adv_scale=self.adv_scale.serialize(), entropy_strength=self.entropy_strength.serialize())
        super().save_config()

    def load_config(self):
        print('load config')
        super().load_config()

        self.policy_lr.load(config=self.config.get('policy_lr', {}))
        self.value_lr.load(config=self.config.get('value_lr', {}))
        self.adv_scale.load(config=self.config.get('adv_scale', {}))
        self.entropy_strength.load(config=self.config.get('entropy_strength', {}))

    def reset(self):
        super().reset()
        self.network.reset()

    def on_episode_end(self):
        super().on_episode_end()
        self.policy_lr.on_episode()
        self.value_lr.on_episode()
        self.adv_scale.on_episode()


class ReinforceMemory:
    def __init__(self, agent: ReinforceAgent):
        self.agent = agent

        if list(self.agent.state_spec.keys()) == ['state']:
            # Simple state-space
            self.states = tf.zeros(shape=(0,) + self.agent.state_spec.get('state'), dtype=tf.float32)
            self.simple_state = True
        else:
            # Complex state-space
            self.states = dict()
            self.simple_state = False

            for name, shape in self.agent.state_spec.items():
                self.states[name] = tf.zeros(shape=(0,) + shape, dtype=tf.float32)

        self.rewards = tf.zeros(shape=(0,), dtype=tf.float32)
        self.values = tf.zeros(shape=(0, 2), dtype=tf.float32)
        self.actions = tf.zeros(shape=(0, self.agent.num_actions), dtype=tf.float32)
        self.log_probabilities = tf.zeros(shape=(0, 1), dtype=tf.float32)
        self.entropy = tf.zeros(shape=(0, 1), dtype=tf.float32)

        # self.returns = None
        # self.advantages = None
        self.returns = tf.zeros(shape=(0, 2), dtype=tf.float32)
        self.advantages = tf.zeros(shape=(0,), dtype=tf.float32)
        self.index = 0

    def append(self, state, action, reward, value):
        if self.simple_state:
            self.states = tf.concat([self.states, state], axis=0)
        else:
            assert isinstance(state, dict)

            for k, v in state.items():
                self.states[k] = tf.concat([self.states[k], v], axis=0)

        self.actions = tf.concat([self.actions, tf.cast(action, dtype=tf.float32)], axis=0)
        self.rewards = tf.concat([self.rewards, [reward]], axis=0)
        self.values = tf.concat([self.values, value], axis=0)

    # def end_trajectory(self, last_value: tf.Tensor):
    #     """Terminates the current trajectory by adding the value of the terminal state"""
    #     value = last_value[:, 0] * tf.pow(10.0, last_value[:, 1])
    #
    #     self.rewards = tf.concat([self.rewards, value], axis=0)
    #     self.values = tf.concat([self.values, last_value], axis=0)

    def end_trajectory(self, last_value: tf.Tensor):
        """Terminates the current trajectory by adding the value of the terminal state"""
        value = last_value[:, 0] * tf.pow(10.0, last_value[:, 1])

        self.rewards = tf.concat([self.rewards, value], axis=0)
        self.values = tf.concat([self.values, last_value], axis=0)

        # compute returns and advantages on last episode only
        new_index = self.index + self.rewards.shape[0]
        rewards = self.rewards[self.index:new_index]
        values = self.values[self.index:new_index]

        returns = self.compute_returns(rewards)
        values, advantages = self.compute_advantages(rewards, values)
        self.index = new_index

        return returns, values, advantages  # just for logging

    # def compute_returns(self, discount: float):
    #     """Computes the returns, also called rewards-to-go"""
    #     returns = utils.rewards_to_go(self.rewards, discount=discount)
    #     returns = utils.to_float(returns)
    #
    #     self.returns = tf.map_fn(fn=utils.decompose_number, elems=returns, dtype=(tf.float32, tf.float32))
    #     self.returns = tf.stack(self.returns, axis=1)
    #     return returns

    def compute_returns(self, rewards):
        """Computes the returns, also called rewards-to-go"""
        returns_ = utils.rewards_to_go(rewards, discount=self.agent.gamma)
        returns_ = utils.to_float(returns_)

        returns = tf.map_fn(fn=utils.decompose_number, elems=returns_, dtype=(tf.float32, tf.float32))
        returns = tf.stack(returns, axis=1)
        self.returns = tf.concat([self.returns, returns], axis=0)

        return returns_

    # def compute_advantages(self, gamma: float, lambda_: float, scale=2.0):
    #     """Computes the advantages using generalized-advantage estimation"""
    #     # value = base * 10^exponent
    #     values = self.values[:, 0] * tf.pow(10.0, self.values[:, 1])
    #
    #     advantages = utils.gae(self.rewards, values=values, gamma=gamma, lambda_=lambda_, normalize=False)
    #     self.advantages = utils.tf_sp_norm(advantages) * scale
    #
    #     return values, advantages

    def compute_advantages(self, rewards, values):
        """Computes the advantages using generalized-advantage estimation"""
        # value = base * 10^exponent
        values_ = values[:, 0] * tf.pow(10.0, values[:, 1])

        advantages = utils.gae(rewards, values=values_, gamma=self.agent.gamma, lambda_=self.agent.lambda_,
                               normalize=False)
        advantages = utils.tf_sp_norm(advantages) * self.agent.adv_scale()
        self.advantages = tf.concat([self.advantages, advantages], axis=0)

        return values_, advantages


# TODO: initializer, try 'glorot_normal', 'he_normal', 'he_uniform', 'orthogonal'
# TODO: kernel/bias regularizer?
# TODO: kernel/bias constraint? - unit_norm or similar?
class Dense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, kernel_initializer='he_normal',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01), **kwargs)


class ReinforceNetwork(Network):

    def __init__(self, agent: ReinforceAgent, policy: dict, value: dict):
        super().__init__(agent)
        self.distribution = self.agent.distribution_type

        # policy and value networks
        self.policy = self.policy_network(**policy)

        self.value = self.value_network(**value)
        self.last_value = tf.zeros((1, 2), dtype=tf.float32)  # (base, exp)

        # keep track of weights for logging
        self._weights = []
        self._biases = []
        self.track_weights()

    @tf.function
    def predict(self, inputs: Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]], actions):
        policy: tfp.distributions.Distribution = self.policy(inputs, training=True)

        if self.distribution == 'categorical':
            if self.agent.num_actions == 1:
                log_prob = policy.log_prob(tf.reshape(actions, shape=[-1]))
                log_prob = tf.expand_dims(log_prob, axis=-1)
            else:
                log_prob = policy.log_prob(actions)
        else:
            log_prob = policy.log_prob(self._clip_actions(actions))

        return log_prob, policy.entropy()

    @tf.function
    def value_predict(self, inputs):
        return self.value(inputs, training=False)

    def predict_last_value(self, state, is_terminal: bool):
        if is_terminal:
            return self.last_value

        return self.value_predict(state)

    @tf.function
    def act(self, inputs: Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]]):
        policy = self.policy(inputs, training=False)
        values = self.value_predict(inputs)

        if self.distribution == 'categorical':
            shape = (tf.shape(policy)[0], self.agent.num_actions)
            action = tf.cast(policy, dtype=tf.float32)
            action = tf.reshape(action, shape)
            mean = std = utils.TF_ZERO
        else:
            action = policy
            mean = policy.mean()
            std = policy.stddev()

        return action, mean, std, values

    def policy_layers(self, inputs: Dict[str, Input], **kwargs):
        """Defines the architecture of the policy-network"""
        units = kwargs.get('units', 32)
        num_layers = kwargs.get('num_layers', kwargs.get('layers', 2))
        activation = kwargs.get('activation', tf.nn.swish)
        dropout_rate = kwargs.get('dropout', 0.0)
        linear_units = kwargs.get('linear_units', 0)

        x = Dense(units, activation=activation)(inputs['state'])
        x = LayerNormalization()(x)

        for _ in range(0, num_layers, 2):
            if dropout_rate > 0.0:
                x = Dense(units, activation=activation)(x)
                x = Dropout(rate=dropout_rate)(x)

                x = Dense(units, activation=activation)(x)
                x = Dropout(rate=dropout_rate)(x)
            else:
                x = Dense(units, activation=activation)(x)
                x = Dense(units, activation=activation)(x)

            x = LayerNormalization()(x)

        if linear_units > 0:
            x = Dense(units=linear_units, activation='linear')(x)

        return x

    def value_layers(self, inputs: Dict[str, Input], **kwargs):
        """Defines the architecture of the value-network"""
        return self.policy_layers(inputs, **kwargs)

    def policy_network(self, **kwargs):
        inputs = self._get_input_layers()
        last_layer = self.policy_layers(inputs, **kwargs)
        actions = self.get_distribution_layer(layer=last_layer)

        return Model(list(inputs.values()), outputs=actions, name='Policy-Network')

    def value_network(self, **kwargs):
        inputs = self._get_input_layers()
        last_layer = self.value_layers(inputs, **kwargs)
        value = self.value_head(last_layer, **kwargs)

        return Model(list(inputs.values()), outputs=value, name='Value-Network')

    def value_head(self, layer: Layer, exponent_scale=6.0, components=1, **kwargs):
        assert components >= 1
        assert exponent_scale > 0.0

        if components == 1:
            base = Dense(units=1, activation=tf.nn.tanh, name='v-base')(layer)
            exp = Dense(units=1, activation=lambda x: exponent_scale * tf.nn.sigmoid(x), name='v-exp')(layer)
        else:
            weights_base = Dense(units=components, activation='softmax', name='w-base')(layer)
            weights_exp = Dense(units=components, activation='softmax', name='w-exp')(layer)

            base = Dense(units=components, activation=tf.nn.tanh, name='v-base')(layer)
            base = utils.tf_dot_product(base, weights_base, axis=1, keepdims=True)

            exp = Dense(units=components, activation=lambda x: exponent_scale * tf.nn.sigmoid(x), name='v-exp')(layer)
            exp = utils.tf_dot_product(exp, weights_exp, axis=1, keepdims=True)

        return concatenate([base, exp], axis=1)

    def get_distribution_layer(self, layer: Layer, index=0) -> tfp.layers.DistributionLambda:
        if self.distribution == 'categorical':
            num_actions = self.agent.num_actions
            num_classes = self.agent.num_classes

            logits = Dense(units=num_actions * num_classes, activation='linear', name='logits')(layer)

            if num_actions > 1:
                logits = Reshape((num_actions, num_classes))(logits)
            # else:
            #     logits = tf.expand_dims(logits, axis=0)

            return tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Categorical(logits=t),
                # convert_to_tensor_fn=self._distribution_to_tensor_categorical
            )(logits)

        if self.distribution == 'beta':
            num_actions = self.agent.num_actions
            alpha = Dense(units=num_actions, activation=utils.softplus(1.0 + 1e-2), name='alpha')(layer)
            beta = Dense(units=num_actions, activation=utils.softplus(1.0 + 1e-2), name='beta')(layer)

            return tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Beta(t[0], t[1]),
                # convert_to_tensor_fn=self._distribution_to_tensor
            )([alpha, beta])

        if self.distribution == 'gaussian':
            num_actions = self.agent.num_actions
            mu = Dense(units=num_actions, activation='linear', name='mu')(layer)
            sigma = Dense(units=num_actions, activation=utils.softplus(1.0 + 1e-2), name='sigma')(layer)

            return tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Normal(loc=t[0], scale=t[1]),
                # convert_to_tensor_fn=self._distribution_to_tensor
            )([mu, sigma])

    @staticmethod
    def _distribution_to_tensor_categorical(d: tfp.distributions.Distribution):
        actions = tf.cast(d.sample(), dtype=tf.float32)
        mean = std = tf.zeros_like(actions)
        return actions, d.log_prob(actions), mean, std, d.entropy()

    def _distribution_to_tensor(self, d: tfp.distributions.Distribution):
        actions = d.sample()
        log_prob = d.log_prob(self._clip_actions(actions))

        return actions, log_prob, d.mean(), d.stddev(), d.entropy()

    # def track_weights(self):
    #     layer = self.policy.get_layer(name='logits')
    #     weights = layer.get_weights()
    #
    #     if len(weights) == 2:
    #         w, b = weights
    #         self._weights.append((f'weight-{layer.name}', w))
    #         self._biases.append((f'bias-{layer.name}', b))
    #
    #     elif len(weights) == 1:
    #         self._weights.append((f'weight-{layer.name}', weights[0]))

    def track_weights(self):
        models = [self.policy, self.value]

        for model in models:
            for layer in model.layers:
                if not layer.trainable:
                    continue

                weights = layer.get_weights()

                if len(weights) == 2:
                    w, b = weights
                    self._weights.append((f'weight-{layer.name}', w))
                    self._biases.append((f'bias-{layer.name}', b))

                elif len(weights) == 1:
                    self._weights.append((f'weight-{layer.name}', weights[0]))

    def log_weights(self):
        for (name, w) in self._weights:
            self.agent.log(**{name: w})

        for (name, b) in self._biases:
            self.agent.log(**{name: b})

    def load_weights(self):
        self.policy.load_weights(filepath=self.agent.weights_path['policy'], by_name=False)
        self.value.load_weights(filepath=self.agent.weights_path['value'], by_name=False)

    def save_weights(self):
        self.policy.save_weights(filepath=self.agent.weights_path['policy'])
        self.value.save_weights(filepath=self.agent.weights_path['value'])

    def summary(self):
        print('==== Policy Network ====')
        self.policy.summary()

        print('\n==== Value Network ====')
        self.value.summary()


if __name__ == '__main__':
    agent = ReinforceAgent(env='CartPole-v0', name='reinforce-cartPole', batch_size=20, drop_batch_remainder=True,
                           lambda_=0.95, optimization_steps=(1, 1), seed=None, log_mode=None)

    # agent.summary()
    # breakpoint()

    agent.learn(episodes=200, timesteps=200, render_every=10, save_every='end')
    pass
