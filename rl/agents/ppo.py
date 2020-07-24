import os
import gym
import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from typing import Union

from rl import utils
from rl.agents.agents import Agent
from rl.parameters import DynamicParameter, ConstantParameter, schedules
from rl.networks.recurrent import OnlineGRU

from tensorflow.keras import losses
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class PPOAgent(Agent):
    # TODO: implement 'action repetition'?
    # TODO: 'value_loss' a parameter that selects the loss (either 'mse' or 'huber') for the value network
    # TODO: 'noise' is broken...
    def __init__(self, *args, policy_lr: Union[float, LearningRateSchedule] = 1e-3, gamma=0.99, lambda_=0.95,
                 value_lr: Union[float, LearningRateSchedule] = 3e-4, optimization_steps=(10, 10), target_kl=False,
                 noise: Union[float, DynamicParameter] = 0.0, clip_ratio: Union[float, DynamicParameter] = 0.2,
                 load=False, name='ppo-agent', entropy_regularization: Union[float, DynamicParameter] = 0.0,
                 recurrence: dict = None, mixture_components=1, clip_norm=(1.0, 1.0), optimizer='adam', **kwargs):
        assert mixture_components >= 1
        super().__init__(*args, name=name, **kwargs)

        self.memory = None
        self.gamma = gamma
        self.lambda_ = lambda_
        self.mixture_components = mixture_components
        self.min_float = tf.constant(np.finfo(np.float32).eps, dtype=tf.float32)

        # Entropy regularization
        if isinstance(entropy_regularization, DynamicParameter):
            self.entropy_strength = entropy_regularization
        else:
            self.entropy_strength = ConstantParameter(entropy_regularization)

        # RNN-related stuff
        if isinstance(recurrence, dict):
            self.is_recurrent = True
            self.rnn_units = recurrence['units']
            self.rnn_depth = recurrence.get('depth', 0)
            self.policy_rnn = None
            self.value_rnn = None
        else:
            self.is_recurrent = False

        # self.drop_batch_reminder |= self.is_recurrent
        self.drop_batch_reminder = False

        # Ratio clipping
        if isinstance(clip_ratio, float):
            assert clip_ratio >= 0.0
            self.clip_ratio = ConstantParameter(value=clip_ratio)
        else:
            assert isinstance(clip_ratio, DynamicParameter)
            self.clip_ratio = clip_ratio

        # KL
        if isinstance(target_kl, float):
            self.early_stop = True
            self.target_kl = tf.constant(target_kl * 1.5)
        else:
            self.early_stop = False

        # TODO: handle complex action spaces
        # Action space
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
                self.num_classes = action_space.nvec[0] + 1  # to include the last class, i.e. 0 to K (not k-1)
                self.convert_action = lambda a: a[0].numpy()
            else:
                self.num_actions = 1
                self.num_classes = action_space.n
                self.convert_action = lambda a: a[0][0].numpy()

        # Gaussian noise (for exploration)
        if isinstance(noise, float):
            self.noise_std = ConstantParameter(value=noise)
        elif isinstance(noise, DynamicParameter):
            self.noise_std = noise
        else:
            raise ValueError("Noise should be an instance of float or DynamicParameter!")

        # print('state_shape:', self.state_shape)
        print('state_spec:', self.state_spec)
        print('action_shape:', self.num_actions)
        print('distribution:', self.distribution_type)

        # Networks & Loading
        self.policy_network = self._policy_network()
        self.value_network = self._value_network()

        if load:
            self.load()

        # Gradient clipping:
        if clip_norm is None:
            self.should_clip_policy_grads = False
            self.should_clip_value_grads = False
        else:
            assert isinstance(clip_norm, tuple)

            if clip_norm[0] is None:
                self.should_clip_policy_grads = False
            else:
                self.should_clip_policy_grads = True
                self.grad_norm_policy = tf.constant(clip_norm[0], dtype=tf.float32)

            if clip_norm[1] is None:
                self.should_clip_value_grads = False
            else:
                self.should_clip_value_grads = True
                self.grad_norm_value = tf.constant(clip_norm[1], dtype=tf.float32)

        # Optimization
        self.policy_optimizer = utils.get_optimizer_by_name(optimizer, learning_rate=policy_lr)
        self.value_optimizer = utils.get_optimizer_by_name(optimizer, learning_rate=value_lr)
        self.optimization_steps = dict(policy=optimization_steps[0], value=optimization_steps[1])

        self.has_schedule_policy = isinstance(policy_lr, schedules.Schedule)
        self.has_schedule_value = isinstance(value_lr, schedules.Schedule)
        self.policy_lr = policy_lr
        self.value_lr = value_lr

        # Incremental mean and std of returns and advantages (used to normalize them)
        self.returns = utils.IncrementalStatistics()
        self.advantages = utils.IncrementalStatistics()

    def act(self, state):
        action = self.policy_network(utils.to_tensor(state), training=False)
        return self.convert_action(action)

    @tf.function
    def predict(self, state):
        policy = self.policy_network(state, training=False)

        if self.distribution_type != 'categorical':
            # round samples (actions) before computing density:
            # motivation: https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Beta
            log_prob = policy.log_prob(tf.clip_by_value(policy, self.min_float, 1.0 - self.min_float))
            mean = policy.mean()
            std = policy.stddev()
        else:
            mean = 0.0
            std = 0.0
            log_prob = policy.log_prob(policy)

        value = self.value_network(state, training=False)

        return policy, mean, std, log_prob, value

    def update(self):
        t0 = time.time()

        # Reset recurrent state before training (to forget what has been seen while acting)
        if self.is_recurrent:
            self.reset_recurrences()

        # Compute advantages and returns:
        advantages = self.get_advantages()
        returns = self.get_returns()

        self.log(returns=returns, advantages=advantages, values=self.memory.values)

        # Prepare data: (states, returns) and (states, advantages, actions, log_prob)
        value_batches = utils.data_to_batches(tensors=(self.memory.states, returns), batch_size=self.batch_size,
                                              drop_remainder=self.drop_batch_reminder, skip=self.skip_count,
                                              map_fn=self.preprocess(),
                                              num_shards=self.obs_skipping, shuffle_batches=self.shuffle_batches)

        policy_batches = utils.data_to_batches(tensors=(self.memory.states, advantages,
                                                        self.memory.actions, self.memory.log_probabilities),
                                               batch_size=self.batch_size, drop_remainder=self.drop_batch_reminder,
                                               skip=self.skip_count, num_shards=self.obs_skipping,
                                               shuffle_batches=self.shuffle_batches, map_fn=self.preprocess())
        # Policy network optimization:
        for opt_step in range(self.optimization_steps['policy']):
            for data_batch in policy_batches:
                with tf.GradientTape() as tape:
                    total_loss, kl = self.ppo_clip_objective(data_batch)

                policy_grads = tape.gradient(total_loss, self.policy_network.trainable_variables)

                if self.should_clip_policy_grads:
                    policy_grads = [tf.clip_by_norm(grad, clip_norm=self.grad_norm_policy) for grad in policy_grads]

                self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy_network.trainable_variables))

                # total_loss, kl, policy_grads = self.update_policy(batch=data_batch)

                self.log(loss_total=total_loss.numpy(),
                         lr_policy=self.policy_lr.lr if self.has_schedule_policy else self.policy_lr,
                         gradients_norm_policy=[tf.norm(gradient) for gradient in policy_grads])

                if self.distribution_type == 'categorical':
                    logits = self.policy_network.get_layer(name='logits')
                    weights, bias = logits.trainable_variables

                    self.log(weights_logits=tf.norm(weights), bias_logits=tf.norm(weights))

                elif self.distribution_type == 'beta':
                    alpha = self.policy_network.get_layer(name='alpha')
                    beta = self.policy_network.get_layer(name='beta')

                    weights_a, bias_a = alpha.trainable_variables
                    weights_b, bias_b = beta.trainable_variables

                    self.log(weights_alpha=tf.norm(weights_a), bias_alpha=tf.norm(bias_a),
                             weights_beta=tf.norm(weights_b), bias_beta=tf.norm(bias_b))

                if self.is_recurrent:
                    self.log(rnn_policy=tf.norm(self.policy_rnn.get_state()))

            # Stop early if target_kl is reached:
            if self.early_stop and (kl > self.target_kl):
                self.log(early_stop=opt_step)
                print(f'early stop at step {opt_step}.')
                break

        # Value network optimization:
        for _ in range(self.optimization_steps['value']):
            for data_batch in value_batches:
                with tf.GradientTape() as tape:
                    value_loss = self.value_objective(batch=data_batch)

                value_grads = tape.gradient(value_loss, self.value_network.trainable_variables)

                if self.should_clip_value_grads:
                    value_grads = [tf.clip_by_norm(grad, clip_norm=self.grad_norm_value) for grad in value_grads]

                self.value_optimizer.apply_gradients(zip(value_grads, self.value_network.trainable_variables))

                # value_loss, value_grads = self.update_value(batch=data_batch)

                self.log(loss_value=value_loss.numpy(),
                         lr_value=self.value_lr.lr if self.has_schedule_value else self.value_lr,
                         gradients_norm_value=[tf.norm(gradient) for gradient in value_grads])

                if self.is_recurrent:
                    self.log(rnn_value=tf.norm(self.value_rnn.get_state()))

        print(f'Update took {round(time.time() - t0, 3)}s')

    # @tf.function
    # def update_policy(self, batch):
    #     """One training step on the policy network"""
    #     with tf.GradientTape() as tape:
    #         total_loss, kl = self.ppo_clip_objective(batch)
    #
    #     policy_grads = tape.gradient(total_loss, self.policy_network.trainable_variables)
    #     self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy_network.trainable_variables))
    #
    #     return total_loss, kl, policy_grads

    # @tf.function
    # def update_value(self, batch):
    #     """One training step on the value network"""
    #     with tf.GradientTape() as tape:
    #         value_loss = self.value_objective(batch=batch)
    #
    #     value_grads = tape.gradient(value_loss, self.value_network.trainable_variables)
    #     self.value_optimizer.apply_gradients(zip(value_grads, self.value_network.trainable_variables))
    #
    #     return value_loss, value_grads

    def predict_value(self, states):
        return self.value_network(states, training=True)

    def value_objective(self, batch):
        states, returns = batch
        values = self.predict_value(states)

        return tf.reduce_mean(losses.mean_squared_error(y_true=returns, y_pred=values))

    def ppo_clip_objective(self, batch):
        states, advantages, actions, old_log_probabilities = batch
        new_policy: tfp.distributions.Distribution = self.policy_network(states, training=True)

        if self.distribution_type == 'categorical' and self.num_actions == 1:
            batch_size = tf.shape(actions)[0]

            new_log_prob = new_policy.log_prob(tf.reshape(actions, shape=batch_size))
            new_log_prob = tf.reshape(new_log_prob, shape=(batch_size, self.num_actions))
        else:
            # round samples (actions) before computing density:
            # motivation: https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Beta
            new_log_prob = new_policy.log_prob(tf.clip_by_value(actions, self.min_float, 1.0 - self.min_float))

        kl_divergence = self.kullback_leibler_divergence(old_log_probabilities, new_log_prob)

        # Entropy
        entropy = new_policy.entropy()
        entropy_coeff = self.entropy_strength()
        entropy_penalty = -entropy_coeff * tf.reduce_mean(entropy)

        # Compute the probability ratio between the current and old policy
        ratio = tf.math.exp(new_log_prob - old_log_probabilities)

        # Compute the clipped ratio times advantage
        clip_value = self.clip_ratio()
        clipped_ratio = tf.clip_by_value(ratio, clip_value_min=1.0 - clip_value, clip_value_max=1.0 + clip_value)

        # Loss = min { ratio * A, clipped_ratio * A } + entropy_term
        policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
        total_loss = policy_loss + entropy_penalty

        # Log stuff
        self.log(ratio=ratio, prob=tf.exp(new_log_prob), entropy=entropy, entropy_coeff=entropy_coeff,
                 ratio_clip=clip_value, kl_divergence=kl_divergence, loss_policy=policy_loss.numpy(),
                 loss_entropy=entropy_penalty.numpy())

        return total_loss, tf.reduce_mean(kl_divergence)

    @staticmethod
    @tf.function
    def kullback_leibler_divergence(log_a, log_b):
        """Source: https://www.tensorflow.org/api_docs/python/tf/keras/losses/KLD"""
        return log_a * (log_a - log_b)

    def get_advantages(self):
        advantages = utils.gae(rewards=self.memory.rewards, values=self.memory.values, gamma=self.gamma,
                               lambda_=self.lambda_, normalize=False)
        self.advantages.update(advantages)
        self.log(advantages_mean=[self.advantages.mean], advantages_std=[self.advantages.std])

        return tf.cast((advantages - self.advantages.mean) / self.advantages.std, dtype=tf.float32)

    def get_returns(self):
        returns = utils.rewards_to_go(rewards=self.memory.rewards, discount=self.gamma, normalize=False)
        self.returns.update(returns)
        self.log(returns_mean=[self.returns.mean], returns_std=[self.returns.std])

        # normalize using running mean and std:
        return (returns - self.returns.mean) / self.returns.std

    def learn(self, episodes: int, timesteps: int, save_every: Union[bool, str, int] = False,
              render_every: Union[bool, str, int] = False):
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
            for episode in range(1, episodes + 1):
                self.reset()
                self.memory = PPOMemory(state_spec=self.state_spec, num_actions=self.num_actions)

                state = self.env.reset()
                state = utils.to_tensor(state)

                # TODO: temporary fix (should be buggy as well...)
                if isinstance(state, dict):
                    state = {f'state_{k}': v for k, v in state.items()}

                episode_reward = 0.0
                t0 = time.time()
                render = episode % render_every == 0

                for t in range(1, timesteps + 1):
                    if render:
                        self.env.render()

                    # Compute action, log_prob, and value
                    action, mean, std, log_prob, value = self.predict(state)
                    action_env = self.convert_action(action)

                    # Environment step
                    next_state, reward, done, _ = self.env.step(action_env)
                    episode_reward += reward

                    self.log(actions=action, action_env=action_env, rewards=reward,
                             distribution_mean=mean, distribution_std=std)

                    self.memory.append(state, action, reward, value, log_prob)
                    state = utils.to_tensor(next_state)

                    if isinstance(state, dict):
                        state = {f'state_{k}': v for k, v in state.items()}

                    # check whether a termination (terminal state or end of a transition) is reached:
                    if done or (t == timesteps):
                        print(f'Episode {episode} terminated after {t} timesteps in {round((time.time() - t0), 3)}s ' +
                              f'with reward {round(episode_reward, 3)}.')
                        self.memory.end_trajectory(last_value=self.last_value if done else self.value_network(state))
                        break

                self.update()
                self.log(episode_rewards=episode_reward)
                self.write_summaries()

                if episode % save_every == 0:
                    self.save()
        finally:
            print('closing...')
            self.env.close()

    def get_distribution_layer(self, layer: Layer) -> tfp.layers.DistributionLambda:
        if self.is_recurrent:
            self.policy_rnn = OnlineGRU(units=self.rnn_units, depth=self.rnn_depth)
            layer = self.policy_rnn(layer)

        if self.distribution_type == 'categorical':
            # Categorical (discrete actions)
            if self.mixture_components == 1:
                logits = Dense(units=self.num_classes * self.num_actions, activation='linear', name='logits')(layer)

                if self.num_actions > 1:
                    logits = Reshape((self.num_actions, self.num_classes))(logits)
                else:
                    logits = tf.expand_dims(logits, axis=0)

                return tfp.layers.DistributionLambda(
                    make_distribution_fn=lambda t: tfp.distributions.Categorical(logits=t))(logits)
            else:
                return utils.get_mixture_of_categorical(layer, num_actions=self.env.action_space.n,
                                                        num_components=self.mixture_components)

        # TODO: poor performance for continuous distributions...
        elif self.distribution_type == 'beta':
            # Beta (bounded continuous 1-dimensional actions)
            # for activations choice refer to chapter 4 of http://proceedings.mlr.press/v70/chou17a/chou17a.pdf

            if self.mixture_components == 1:
                # make a, b > 1, so that the Beta distribution is concave and unimodal (see paper above)
                # alpha = Dense(units=self.num_actions, activation=utils.softplus_one, name='alpha')(layer)
                # beta = Dense(units=self.num_actions, activation=utils.softplus_one, name='beta')(layer)

                alpha = Dense(units=self.num_actions, activation='softplus', name='alpha')(layer)
                beta = Dense(units=self.num_actions, activation='softplus', name='beta')(layer)

                return tfp.layers.DistributionLambda(
                    make_distribution_fn=lambda t: tfp.distributions.Beta(t[0], t[1]))([alpha, beta])
            else:
                return utils.get_mixture_of_beta(layer, self.num_actions, num_components=self.mixture_components)

        # Gaussian (unbounded continuous actions)
        # for activations choice see chapter 4 of http://proceedings.mlr.press/v70/chou17a/chou17a.pdf
        if self.mixture_components == 1:
            mu = Dense(units=self.num_actions, activation='linear', name='mu')(layer)
            sigma = Dense(units=self.num_actions, activation='softplus', name='sigma')(layer)

            # ensure variance > 0, so that loss doesn't diverge or became NaN
            sigma = tf.add(sigma, self.min_float)

            return tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Normal(loc=t[0], scale=t[1])
            )([mu, sigma])
        else:
            return utils.get_mixture_of_gaussian(layer, self.num_actions, num_components=self.mixture_components)

    def policy_layers(self, inputs: dict, **kwargs) -> Layer:
        """Main (central) part of the policy network"""
        units = kwargs.get('units', 32)
        num_layers = kwargs.get('layers', 2)
        activation = kwargs.get('activation', 'swish')
        dropout_rate = kwargs.get('dropout', 0.0)

        x = Dense(units, activation=activation)(inputs['state'])
        x = LayerNormalization()(x)

        for _ in range(0, num_layers, 2):
            x = Dense(units, activation=activation)(x)
            x = Dropout(rate=dropout_rate)(x)

            x = Dense(units, activation=activation)(x)
            x = Dropout(rate=dropout_rate)(x)

            x = LayerNormalization()(x)

        return x

    def value_layers(self, inputs: dict, **kwargs) -> Layer:
        """Main (central) part of the value network"""
        # Default: use same layers as policy
        return self.policy_layers(inputs, **kwargs)

    def _policy_network(self) -> Model:
        inputs = self._get_input_layers()
        x = self.policy_layers(inputs)
        action = self.get_distribution_layer(layer=x)

        return Model(inputs=list(inputs.values()), outputs=action, name='policy')

    def _value_network(self) -> Model:
        """Single-head Value Network"""
        inputs = self._get_input_layers()
        x = self.value_layers(inputs)

        if self.is_recurrent:
            self.value_rnn = OnlineGRU(units=self.rnn_units, depth=self.rnn_depth)
            x = self.value_rnn(x)

        value = Dense(units=1, activation=None, dtype=tf.float32, name='value_head')(x)

        return Model(inputs=list(inputs.values()), outputs=value, name='value_network')

    def summary(self):
        self.policy_network.summary()
        self.value_network.summary()

    def save_weights(self):
        print('saving weights...')
        self.policy_network.save_weights(filepath=self.weights_path['policy'])
        self.value_network.save_weights(filepath=self.weights_path['value'])

    def load_weights(self):
        print('loading weights...')
        self.policy_network.load_weights(filepath=self.weights_path['policy'], by_name=False)
        self.value_network.load_weights(filepath=self.weights_path['value'], by_name=False)

    def save_config(self):
        print('save config')
        self.update_config(returns=self.returns.as_dict(), advantages=self.advantages.as_dict())
        super().save_config()
    
    def load_config(self):
        print('load config')
        super().load_config()
        self.returns.set(**self.config['returns'])
        self.advantages.set(**self.config['advantages'])

    def reset(self):
        super().reset()

        if self.is_recurrent:
            self.reset_recurrences()

    def reset_recurrences(self):
        """Resets both policy and value network's RNNs state"""
        self.reset_policy_rnn()
        self.reset_value_rnn()

    def reset_policy_rnn(self):
        """Resets the value network's RNNs state"""
        self.policy_rnn.reset_state()

    def reset_value_rnn(self):
        """Resets the value network's RNNs state"""
        self.value_rnn.reset_state()


class PPOMemory:
    """Recent memory used in PPOAgent"""
    def __init__(self, state_spec: dict, num_actions: int):
        if list(state_spec.keys()) == ['state']:
            # Simple state-space
            self.states = tf.zeros(shape=(0,) + state_spec.get('state'), dtype=tf.float32)
            self.simple_state = True
        else:
            # Complex state-space
            self.states = dict()
            self.simple_state = False

            for name, shape in state_spec.items():
                self.states[name] = tf.zeros(shape=(0,) + shape, dtype=tf.float32)

        self.rewards = tf.zeros(shape=(0,), dtype=tf.float32)
        self.values = tf.zeros(shape=(0, 1), dtype=tf.float32)
        self.actions = tf.zeros(shape=(0, num_actions), dtype=tf.float32)
        self.log_probabilities = tf.zeros(shape=(0, num_actions), dtype=tf.float32)

    def append(self, state, action, reward, value, log_prob):
        if self.simple_state:
            self.states = tf.concat([self.states, state], axis=0)
        else:
            assert isinstance(state, dict)

            for k, v in state.items():
                self.states[k] = tf.concat([self.states[k], v], axis=0)

        self.actions = tf.concat([self.actions, tf.cast(action, dtype=tf.float32)], axis=0)
        self.rewards = tf.concat([self.rewards, [reward]], axis=0)
        self.values = tf.concat([self.values, value], axis=0)

        # double hack: `tf.reshape` for Categorical, and `tf.reduce_mean` for Beta
        self.log_probabilities = tf.concat([self.log_probabilities, log_prob], axis=0)

    def end_trajectory(self, last_value):
        """Terminates the current trajectory by adding the value of the terminal state"""
        self.rewards = tf.concat([self.rewards, last_value[0]], axis=0)
        self.values = tf.concat([self.values, last_value], axis=0)
