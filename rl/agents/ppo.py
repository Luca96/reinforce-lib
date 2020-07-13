import os
import gym
import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from typing import Union

from rl import utils
from rl.agents.agents import Agent
from rl.parameters import DynamicParameter, ConstantParameter

from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class PPOAgent(Agent):
    # TODO: implement 'action repetition'?
    # TODO: 'value_loss' a parameter that selects the loss (either 'mse' or 'huber') for the value network
    # TODO: try 'mixture' of Beta/Gaussian distribution
    # TODO: make 'clip_ratio', 'entropy_reg' as dynamic parameters...
    # TODO: 'noise' is broken...
    def __init__(self, *args, policy_lr: Union[float, LearningRateSchedule] = 1e-3, clip_ratio=0.2, gamma=0.99,
                 value_lr: Union[float, LearningRateSchedule] = 3e-4, optimization_steps=(10, 10), lambda_=0.95,
                 noise: Union[float, DynamicParameter] = 0.0, target_kl=False, entropy_regularization=0.0, load=False,
                 name='ppo-agent', recurrent_policy=False, recurrent_units=8, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.memory = None
        self.gamma = gamma
        self.lambda_ = lambda_
        self.entropy_strength = tf.constant(entropy_regularization)
        self.is_recurrent = recurrent_policy
        self.recurrent_units = recurrent_units
        self.drop_batch_reminder |= self.is_recurrent

        # Ratio clipping
        self.min_ratio = tf.constant(1.0 - clip_ratio)
        self.max_ratio = tf.constant(1.0 + clip_ratio)

        if isinstance(target_kl, float):
            self.early_stop = True
            self.target_kl = tf.constant(target_kl * 1.5)
        else:
            self.early_stop = False

        # TODO: handle complex action spaces
        # Action space
        if isinstance(self.env.action_space, gym.spaces.Box):
            self.num_actions = self.env.action_space.shape[0]

            # continuous:
            if self.env.action_space.is_bounded():
                self.distribution_type = 'beta'
                self.action_low = tf.constant(self.env.action_space.low, dtype=tf.float32)
                self.action_range = tf.constant(self.env.action_space.high - self.env.action_space.low,
                                                dtype=tf.float32)
                self.convert_action = lambda a: (a * self.action_range + self.action_low)[0].numpy()
            else:
                self.distribution_type = 'gaussian'
                self.normalize_action = lambda a: a[0].numpy()
        else:
            # discrete:
            self.num_actions = 1
            self.distribution_type = 'categorical'
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
            self.load_weights()

        # Optimization
        self.policy_optimizer = optimizers.Adam(learning_rate=policy_lr)
        self.value_optimizer = optimizers.Adam(learning_rate=value_lr)
        self.optimization_steps = dict(policy=optimization_steps[0], value=optimization_steps[1])

        # Incremental mean and std of "returns" (used to normalize them)
        self.returns_mean = 0.0
        self.returns_variance = 0.0
        self.returns_std = 0.0
        self.returns_count = 0

        # Incremental mean and std of "advantages" (used to normalize them)
        self.adv_mean = 0.0
        self.adv_variance = 0.0
        self.adv_std = 0.0
        self.adv_count = 0

    def act(self, state):
        action = self.policy_network(utils.to_tensor(state), training=False)
        return self.convert_action(action)

    def update(self):
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
                    policy_loss, kl = self.ppo_clip_objective(data_batch)

                policy_grads = tape.gradient(policy_loss, self.policy_network.trainable_variables)
                self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy_network.trainable_variables))

                self.log(loss_policy=policy_loss.numpy(),
                         gradients_norm_policy=[tf.norm(gradient) for gradient in policy_grads])

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
                self.value_optimizer.apply_gradients(zip(value_grads, self.value_network.trainable_variables))

                self.log(loss_value=value_loss.numpy(),
                         gradients_norm_value=[tf.norm(gradient) for gradient in value_grads])
        
        if self.is_recurrent:
            self.policy_network.reset_states()
            self.value_network.reset_states()

    def value_objective(self, batch):
        states, returns = batch
        values = self.value_network(states, training=True)

        return tf.reduce_mean(losses.mean_squared_error(y_true=returns, y_pred=values))

    def ppo_clip_objective(self, batch):
        states, advantages, actions, old_log_probabilities = batch
        new_policy: tfp.distributions.Distribution = self.policy_network(states, training=True)

        new_log_prob = new_policy.log_prob(actions)

        # Shape (or better, 'broadcast') fix.
        # NOTE: this "magically" makes everything works fine...
        if new_log_prob.shape != old_log_probabilities.shape:
            new_log_prob = tf.reduce_mean(new_log_prob, axis=0)
            new_log_prob = tf.expand_dims(new_log_prob, axis=-1)

        kl_divergence = self.kullback_leibler_divergence(old_log_probabilities, new_log_prob)

        # Entropy
        entropy = new_policy.entropy()
        # entropy_term = self.entropy_strength * entropy
        entropy_penalty = -self.entropy_strength * tf.reduce_mean(entropy)

        # Compute the probability ratio between the current and old policy
        ratio = tf.math.exp(new_log_prob - old_log_probabilities)

        # Compute the clipped ratio times advantage (NOTE: this is the simplified PPO clip-objective):
        # clipped_ratio = tf.where(advantages > 0, x=self.max_ratio, y=self.min_ratio)
        clipped_ratio = tf.clip_by_value(ratio, self.min_ratio, self.max_ratio)

        # Log stuff
        self.log(ratio=ratio, prob=tf.exp(new_log_prob), entropy=entropy, kl_divergence=kl_divergence)

        # Loss = min { ratio * A, clipped_ratio * A } + entropy_term
        # loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages) + entropy_term)

        policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
        total_loss = policy_loss + entropy_penalty

        return total_loss, tf.reduce_mean(kl_divergence)

    @staticmethod
    def kullback_leibler_divergence(log_a, log_b):
        """Source: https://www.tensorflow.org/api_docs/python/tf/keras/losses/KLD"""
        return log_a * (log_a - log_b)

    def get_advantages(self):
        advantages = utils.gae(rewards=self.memory.rewards, values=self.memory.values, gamma=self.gamma,
                               lambda_=self.lambda_, normalize=False)
        self._update_adv_mean_and_std(advantages)

        return tf.cast((advantages - self.adv_mean) / self.adv_std, dtype=tf.float32)

    def get_returns(self):
        returns = utils.rewards_to_go(rewards=self.memory.rewards, discount=self.gamma, normalize=False)
        self._update_returns_mean_and_std(returns)

        # normalize using running mean and std:
        return (returns - self.returns_mean) / self.returns_std

    def learn(self, episodes: int, timesteps: int, save_every: Union[bool, str, int] = False,
              render_every: Union[bool, str, int] = False):
        self.clear()

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
                    policy = self.policy_network(state, training=False)
                    action = policy
                    log_prob = policy.log_prob(action)
                    value = self.value_network(state, training=False)
                    action_env = self.convert_action(action)

                    # Environment step
                    next_state, reward, done, _ = self.env.step(action_env)
                    episode_reward += reward

                    self.log(actions=action, action_env=action_env, rewards=reward)

                    self.memory.append(state, action, reward, value, log_prob)
                    state = utils.to_tensor(next_state)

                    if isinstance(state, dict):
                        state = {f'state_{k}': v for k, v in state.items()}

                    # check whether a termination (terminal state or end of a transition) is reached:
                    if done or (t == timesteps):
                        print(f'Episode {episode} terminated after {t} timesteps in {round((time.time() - t0), 3)}s ' +
                              f'with reward {episode_reward}.')
                        self.memory.end_trajectory(last_value=self.last_value if done else self.value_network(state))
                        break

                self.update()
                self.log(episode_rewards=episode_reward)
                self.write_summaries()

                if episode % save_every == 0:
                    self.save_weights()
        finally:
            print('closing...')
            self.env.close()

    def get_distribution_layer(self, layer: Layer) -> tfp.layers.DistributionLambda:
        if self.distribution_type == 'categorical':
            # Categorical
            if self.is_recurrent:
                layer = tf.expand_dims(layer, axis=0)
                layer = GRU(units=self.recurrent_units, stateful=True)(layer)

            logits = Dense(units=self.env.action_space.n, activation='linear')(layer)
            logits = tf.expand_dims(logits, axis=0, name='logits')

            return tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Categorical(logits=t))(logits)

        elif self.distribution_type == 'beta':
            # Beta
            # for activations choice see chapter 4 of http://proceedings.mlr.press/v70/chou17a/chou17a.pdf
            if self.is_recurrent:
                layer = tf.expand_dims(layer, axis=0)
                layer = GRU(units=self.recurrent_units, stateful=True)(layer)

            alpha = Dense(units=self.num_actions, activation='softplus')(layer)
            beta = Dense(units=self.num_actions, activation='softplus')(layer)

            alpha = Add(name='alpha')([alpha, tf.ones_like(alpha)])
            beta = Add(name='beta')([beta, tf.ones_like(beta)])

            return tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Beta(t[0], t[1]))([alpha, beta])

        # Gaussian (Normal)
        # for activations choice see chapter 4 of http://proceedings.mlr.press/v70/chou17a/chou17a.pdf
        if self.is_recurrent:
            layer = tf.expand_dims(layer, axis=0)
            layer = GRU(units=self.recurrent_units, stateful=True)(layer)

        mu = Dense(units=self.num_actions, activation='linear', name='mu')(layer)
        sigma = Dense(units=self.num_actions, activation='softplus', name='sigma')(layer)

        return tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfp.distributions.Normal(loc=t[0], scale=t[1]))([mu, sigma])

    def policy_layers(self, inputs: dict, **kwargs) -> Layer:
        """Main (central) part of the policy network"""
        units = kwargs.get('units', 32)
        num_layers = kwargs.get('layers', 2)

        x = Dense(units, activation='tanh')(inputs['state'])

        for _ in range(num_layers):
            x = Dense(units, activation='relu')(x)
            x = Dense(units, activation='relu')(x)

        return x

    def value_layers(self, inputs: dict, **kwargs) -> Layer:
        """Main (central) part of the value network"""
        # Default: use same layers as policy
        return self.policy_layers(inputs)

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
            x = tf.expand_dims(x, axis=0)
            x = GRU(units=self.recurrent_units, stateful=True)(x)

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

    def clear(self):
        # Reset incremental statistics for returns and advantages:
        self.returns_mean = 0.0
        self.returns_variance = 0.0
        self.returns_std = 0.0
        self.returns_count = 0

        self.adv_mean = 0.0
        self.adv_variance = 0.0
        self.adv_std = 0.0
        self.adv_count = 0

    def _update_returns_mean_and_std(self, returns):
        """Calculates the incremental mean and standard deviation of returns. one batch at a time.
        Sources:
          (1) http://datagenetics.com/blog/november22017/index.html
          (2) https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
        """
        old_mean = self.returns_mean
        ret_mean = tf.reduce_mean(returns)
        m = self.returns_count
        n = tf.shape(returns)[0]
        c1 = m / (m + n)
        c2 = n / (m + n)
        c3 = (m * n) / (m + n) ** 2

        self.returns_mean = c1 * old_mean + c2 * ret_mean
        self.returns_variance = c1 * self.returns_variance + c2 * tf.math.reduce_variance(returns) + \
                                c3 * (old_mean - ret_mean) ** 2

        self.returns_std = tf.sqrt(self.returns_variance)
        self.returns_count += n

        self.log(returns_mean=[self.returns_mean], returns_std=[self.returns_std])

    def _update_adv_mean_and_std(self, advantages):
        """Calculates the incremental mean and standard deviation of returns. one batch at a time."""
        old_mean = self.adv_mean
        adv_mean = tf.reduce_mean(advantages)
        m = self.adv_count
        n = tf.shape(advantages)[0]
        c1 = m / (m + n)
        c2 = n / (m + n)
        c3 = (m * n) / (m + n) ** 2

        self.adv_mean = c1 * old_mean + c2 * adv_mean
        self.adv_variance = c1 * self.adv_variance + c2 * tf.math.reduce_variance(advantages) + \
                            c3 * (old_mean - adv_mean) ** 2

        self.adv_std = tf.sqrt(self.adv_variance)
        self.adv_count += n

        self.log(advantages_mean=[self.adv_mean], advantages_std=[self.adv_std])

    # def _load_distribution_layer(self, model):
    #     """This function fixes the fact that keras.models.load_model does not return a distribution object,
    #         but a tensor instead.
    #     """
    #     if self.distribution_type == 'categorical':
    #         logits = model.get_layer(name='logits')
    #         logits = self._add_gaussian_noise(logits.output)
    #
    #         action = tfp.layers.DistributionLambda(
    #             make_distribution_fn=lambda t: tfp.distributions.Categorical(logits=t))(logits)
    #
    #     elif self.distribution_type == 'beta':
    #         alpha = model.get_layer(name='alpha')
    #         beta = model.get_layer(name='beta')
    #
    #         alpha = self._add_gaussian_noise(alpha.output)
    #         alpha = tf.nn.softplus(alpha) + tf.ones_like(alpha)
    #
    #         beta = self._add_gaussian_noise(beta.output)
    #         beta = tf.nn.softplus(beta) + tf.ones_like(beta)
    #
    #         action = tfp.layers.DistributionLambda(
    #             make_distribution_fn=lambda t: tfp.distributions.Beta(t[0], t[1]))([alpha, beta])
    #     else:
    #         # Gaussian
    #         mu = model.get_layer(name='mu')
    #         sigma = model.get_layer(name='sigma')
    #
    #         sigma = self._add_gaussian_noise(sigma.output)
    #         sigma = tf.nn.softplus(sigma)
    #
    #         action = tfp.layers.DistributionLambda(
    #             make_distribution_fn=lambda t: tfp.distributions.Normal(loc=t[0], scale=t[1]))([mu.output, sigma])
    #
    #     return Model(inputs=model.input, outputs=action, name='policy')


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
        self.log_probabilities = tf.concat([self.log_probabilities, log_prob], axis=0)

    def end_trajectory(self, last_value):
        """Terminates the current trajectory by adding the value of the terminal state"""
        self.rewards = tf.concat([self.rewards, last_value[0]], axis=0)
        self.values = tf.concat([self.values, last_value], axis=0)
