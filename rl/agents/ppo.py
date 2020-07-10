import os
import gym
import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from typing import Union

from rl import utils
from rl.agents.agents import Agent
from rl.exploration import RandomNetworkDistillation, NoExploration

from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


class PPOAgent(Agent):
    # TODO: same 'optimization steps' for both policy and value functions?
    # TODO: implement 'action repetition'?
    # TODO: 'value_loss' a parameter that selects the loss (either 'mse' or 'huber') for the value network
    # TODO: try 'mixture' of Beta/Gaussian distribution
    #  https://keras.io/guides/training_with_built_in_methods/#using-learning-rate-schedules
    # TODO: decay and/or halve parameters (lr, noise, ...) by step
    def __init__(self, *args, policy_lr=3e-4, value_lr=1e-4, optimization_steps=(10, 10), clip_ratio=0.2, noise=0.0,
                 gamma=0.99, lambda_=0.95, target_kl=0.01, entropy_regularization=0.0, load=False, name='ppo-agent',
                 **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.memory = None
        self.gamma = gamma
        self.lambda_ = lambda_
        self.entropy_strength = tf.constant(entropy_regularization)

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
                self.action_range = self.env.action_space.high - self.env.action_space.low
                self.convert_action = lambda a: a * self.action_range + self.env.action_space.low
            else:
                self.distribution_type = 'gaussian'
                self.convert_action = lambda a: a

            # Gaussian noise (for exploration)
            if noise > 0.0:
                self.add_noise = lambda a: a + tf.random.normal(a.shape, mean=0.0, stddev=noise)
            else:
                self.add_noise = lambda a: a
        else:
            # discrete:
            self.num_actions = 1
            self.distribution_type = 'categorical'
            self.convert_action = lambda a: a
            self.add_noise = lambda a: a

        # print('state_shape:', self.state_shape)
        print('state_spec:', self.state_spec)
        print('action_shape:', self.num_actions)
        print('distribution:', self.distribution_type)

        # Networks
        if load:
            self.load()
        else:
            self.policy_network = self._policy_network()
            self.value_network = self._value_network()

        # Optimization
        self.policy_optimizer = optimizers.Adam(learning_rate=policy_lr)
        self.value_optimizer = optimizers.Adam(learning_rate=value_lr)
        self.optimization_steps = dict(policy=optimization_steps[0], value=optimization_steps[1])

    def act(self, state):
        action = self.policy_network(utils.to_tensor(state), training=False)
        return self.convert_action(action[0][0].numpy())

    def update(self):
        # Compute combined advantages and returns:
        advantages = self.get_advantages()
        returns = utils.rewards_to_go(rewards=self.memory.rewards, discount=self.gamma, normalize=True)

        # Log
        self.log(returns=returns, advantages=advantages, values=self.memory.values)

        # Prepare data: (states, returns) and (states, advantages, actions, log_prob)
        value_batches = utils.data_to_batches(tensors=(self.memory.states, returns), batch_size=self.batch_size,
                                              drop_remainder=self.drop_batch_reminder, skip=self.skip_count,
                                              num_shards=self.obs_skipping, shuffle_batches=self.shuffle_batches)

        policy_batches = utils.data_to_batches(tensors=(self.memory.states, advantages,
                                                        self.memory.actions, self.memory.log_probabilities),
                                               batch_size=self.batch_size, drop_remainder=self.drop_batch_reminder,
                                               skip=self.skip_count, num_shards=self.obs_skipping,
                                               shuffle_batches=self.shuffle_batches)
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

    def value_objective(self, batch):
        states, returns = batch
        values = self.value_network(states, training=True)

        return tf.reduce_mean(losses.mean_squared_error(y_true=returns, y_pred=values))

    def ppo_clip_objective(self, batch):
        states, advantages, actions, old_log_probabilities = batch
        new_policy: tfp.distributions.Distribution = self.policy_network(states, training=True)

        new_log_prob = new_policy.log_prob(actions)

        # Shape (or better, 'broadcast') fix:
        if new_log_prob.shape != old_log_probabilities.shape:
            new_log_prob = tf.reduce_mean(new_log_prob, axis=0)
            new_log_prob = tf.expand_dims(new_log_prob, axis=-1)

        kl_divergence = self.kullback_leibler_divergence(old_log_probabilities, new_log_prob)

        # Entropy
        entropy = new_policy.entropy()
        entropy_term = self.entropy_strength * entropy

        # Compute the probability ratio between the current and old policy
        ratio = tf.math.exp(new_log_prob - old_log_probabilities)

        # Compute the clipped ratio times advantage (NOTE: this is the simplified PPO clip-objective):
        clipped_ratio = tf.where(advantages > 0, x=self.max_ratio, y=self.min_ratio)

        # Log stuff
        self.log(ratio=ratio, prob=tf.exp(new_log_prob), entropy=entropy, kl_divergence=kl_divergence)

        # Loss = min { ratio * A, clipped_ratio * A } + entropy_term
        loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages) + entropy_term)
        return loss, tf.reduce_mean(kl_divergence)

    @staticmethod
    def kullback_leibler_divergence(log_a, log_b):
        """Source: https://www.tensorflow.org/api_docs/python/tf/keras/losses/KLD"""
        return log_a * (log_a - log_b)

    def get_advantages(self):
        return utils.gae(rewards=self.memory.rewards, values=self.memory.values,
                         gamma=self.gamma, lambda_=self.lambda_, normalize=True)

    def learn(self, episodes: int, timesteps: int, save_every: Union[bool, str, int] = False,
              render_every=0):
        if save_every is False:
            save_every = episodes + 1
        elif save_every is True:
            save_every = 1
        elif save_every == 'end':
            save_every = episodes

        for episode in range(1, episodes + 1):
            self.memory = PPOMemory(capacity=timesteps, state_spec=self.state_spec, num_actions=self.num_actions)
            state = self.env.reset()
            state = utils.to_tensor(state)
            episode_reward = 0.0
            t0 = time.time()
            render = episode % render_every == 0

            for t in range(1, timesteps + 1):
                if render:
                    self.env.render()

                # Compute action, log_prob, and value
                policy = self.policy_network(state, training=False)
                action = self.add_noise(policy)
                log_prob = policy.log_prob(action)
                value = self.value_network(state, training=False)

                # Make action in the right range for the environment
                converted_action = self.convert_action(action[0][0].numpy())

                # Environment step
                next_state, reward, done, _ = self.env.step(converted_action)
                episode_reward += reward

                self.log(actions=converted_action, rewards=reward)

                self.memory.append(state, action, reward, value, log_prob)
                state = utils.to_tensor(next_state)

                # check whether a termination (terminal state or end of a transition) is reached:
                if done or (t == timesteps):
                    print(f'Episode {episode} terminated after {t} timesteps in {round((time.time() - t0), 4)}s ' +
                          f'with reward {episode_reward}.')
                    self.memory.end_trajectory(last_value=0 if done else self.value_network(state)[0])
                    break

            self.update()
            self.log(episode_rewards=episode_reward)
            self.write_summaries()

            if episode % save_every == 0:
                self.save()

        self.env.close()

    def get_distribution_layer(self, layer: Layer) -> tfp.layers.DistributionLambda:
        if self.distribution_type == 'categorical':
            # Categorical
            logits = Dense(units=self.env.action_space.n, activation=None, name='logits')(layer)

            return tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Categorical(logits=t),
                convert_to_tensor_fn=lambda s: s.sample(self.num_actions))(logits)

        elif self.distribution_type == 'beta':
            # Beta
            # for activations choice see chapter 4 of http://proceedings.mlr.press/v70/chou17a/chou17a.pdf
            alpha = Dense(units=self.num_actions, activation='softplus')(layer)
            alpha = Add(name='alpha')([alpha, tf.ones_like(alpha)])

            beta = Dense(units=self.num_actions, activation='softplus')(layer)
            beta = Add(name='beta')([beta, tf.ones_like(beta)])

            return tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Beta(t[0], t[1]),
                convert_to_tensor_fn=lambda s: s.sample(self.num_actions))([alpha, beta])

        # Gaussian (Normal)
        # for activations choice see chapter 4 of http://proceedings.mlr.press/v70/chou17a/chou17a.pdf
        mu = Dense(units=self.num_actions, activation='linear', name='mu')(layer)
        sigma = Dense(units=self.num_actions, activation='softplus', name='sigma')(layer)

        return tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfp.distributions.Normal(loc=t[0], scale=t[1]),
            convert_to_tensor_fn=lambda s: s.sample(self.num_actions))([mu, sigma])

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
        value = Dense(units=1, activation=None, name='value_head')(x)

        return Model(inputs=list(inputs.values()), outputs=value, name='value_network')

    def summary(self):
        self.policy_network.summary()
        self.value_network.summary()

    def save(self):
        print('saving...')
        self.policy_network.save(self.save_path['policy'], include_optimizer=False)
        self.value_network.save(self.save_path['value'], include_optimizer=False)

    def load(self):
        # TODO: when loading the model save: iteration number (for logs), dynamic parameters (e.g. learning rate) ..
        print('loading...')
        model = tf.keras.models.load_model(self.save_path['policy'], compile=False)
        self.value_network = tf.keras.models.load_model(self.save_path['value'], compile=False)

        # when loading substitute the 'action' layer with a tfp.distribution instance!
        self.policy_network = self._load_distribution_layer(model)

    def _load_distribution_layer(self, model):
        """This function fixes the fact that keras.models.load_model does not return a distribution object,
            but a tensor instead.
        """
        if self.distribution_type == 'categorical':
            logits = model.get_layer(name='logits')

            action = tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Categorical(logits=t),
                convert_to_tensor_fn=lambda s: s.sample(self.num_actions))(logits.output)

        elif self.distribution_type == 'beta':
            alpha = model.get_layer(name='alpha')
            beta = model.get_layer(name='beta')

            action = tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Beta(t[0], t[1]),
                convert_to_tensor_fn=lambda s: s.sample(self.num_actions))([alpha.output, beta.output])
        else:
            # Gaussian
            mu = model.get_layer(name='mu')
            sigma = model.get_layer(name='sigma')

            action = tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Normal(loc=t[0], scale=t[1]),
                convert_to_tensor_fn=lambda s: s.sample(self.num_actions))([mu.output, sigma.output])

        return Model(inputs=model.input, outputs=action, name='policy')


class PPOMemory:
    def __init__(self, capacity: int, state_spec: dict, num_actions: int):
        self.index = 0
        self.size = capacity

        if list(state_spec.keys()) == ['state']:
            # Simple state-space
            self.states = np.zeros(shape=(capacity,) + state_spec.get('state'), dtype=np.float32)
            self.simple_state = True
        else:
            # Complex state-space
            self.states = dict()
            self.simple_state = False

            for name, shape in state_spec.items():
                self.states[name] = np.zeros(shape=(capacity,) + shape, dtype=np.float32)

        self.rewards = np.zeros(shape=capacity + 1, dtype=np.float32)
        self.values = np.zeros(shape=capacity + 1, dtype=np.float32)
        self.actions = np.zeros(shape=(capacity, num_actions), dtype=np.float32)
        self.log_probabilities = np.zeros(shape=(capacity, 1), dtype=np.float32)

    def append(self, state, action, reward, value, log_prob):
        assert self.index < self.size
        i = self.index

        if self.simple_state:
            self.states[i] = tf.squeeze(state)
        else:
            assert isinstance(state, dict)

            for k, v in state.items():
                self.states[f'{state}_{k}'][i] = tf.squeeze(v)

        self.actions[i] = tf.squeeze(action)
        self.rewards[i] = reward
        self.values[i] = utils.tf_to_scalar_shape(value)
        self.log_probabilities[i] = log_prob
        self.index += 1

    def end_trajectory(self, last_value):
        """Terminates the current trajectory by adding the value of the terminal state"""
        self.rewards[self.index] = last_value
        self.values[self.index] = last_value

        # cut off the exceeding part
        if self.index < self.size:
            self.actions = self.actions[:self.index]
            self.rewards = self.rewards[:self.index + 1]
            self.values = self.values[:self.index + 1]
            self.log_probabilities = self.log_probabilities[:self.index]

            if self.simple_state:
                self.states = self.states[:self.index]
            else:
                for k, v in self.states.items():
                    self.states[k] = v[:self.index]


# -------------------------------------------------------------------------------------------------
# -- PPO2: PPO Agent with RND exploration method
# -------------------------------------------------------------------------------------------------

# TODO: must update
class PPO2Agent(PPOAgent):
    """Proximal Policy Optimization (PPO) agent with Random Network Distillation (RND) exploration method """
    def __init__(self, *args, advantage_weights=(0.5, 0.5), **kwargs):
        super().__init__(*args, **kwargs)

        # Exploration
        self.exploration = RandomNetworkDistillation(state_shape=self.state_shape, reward_shape=1,
                                                     batch_size=32, optimization_steps=1, num_layers=1)
        self.adv_weights = (tf.constant(advantage_weights[0]), tf.constant(advantage_weights[1]))
        self.value_coeff = (tf.constant(0.5), tf.constant(0.5))

        # Statistics
        # self.stats = dict(loss_policy=[[], 0], loss_value=[[], 0], episode_rewards=[[], 0], ratio=[[], 0],
        #                   advantages=[[], 0], prob=[[], 0], actions=[[], 0],
        #                   entropy=[[], 0], kl_divergence=[[], 0], rewards_extrinsic=[[], 0],
        #                   rewards_intrinsic=[[], 0], values_intrinsic=[[], 0], values_extrinsic=[[], 0],
        #                   returns_extrinsic=[[], 0], returns_intrinsic=[[], 0],
        #                   advantages_intrinsic=[[], 0], advantages_extrinsic=[[], 0],
        #                   gradients_norm_policy=[[], 0], gradients_norm_value=[[], 0])

    def update(self, batch_size: int):
        # Compute combined advantages and returns:
        advantages = self.get_advantages()
        extrinsic_returns, intrinsic_returns = self.get_returns()

        # Log
        self.log(returns_extrinsic=extrinsic_returns, returns_intrinsic=intrinsic_returns,
                 advantages=advantages,
                 values_extrinsic=self.memory.extrinsic_values, values_intrinsic=self.memory.intrinsic_values)

        # Prepare data: (states, returns) and (states, advantages)
        value_batches = utils.data_to_batches(tensors=(self.memory.states, extrinsic_returns, intrinsic_returns),
                                              batch_size=batch_size)
        policy_batches = utils.data_to_batches(tensors=(self.memory.states, advantages,
                                                        self.memory.actions, self.memory.log_probabilities),
                                               batch_size=batch_size)

        # Policy network optimization:
        # TODO: here we ignore 'optimization_stes' parameter...
        for step, data_batch in enumerate(policy_batches):
            with tf.GradientTape() as tape:
                policy_loss, kl = self.ppo_clip_objective(data_batch)

            policy_grads = tape.gradient(policy_loss, self.policy_network.trainable_variables)
            self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy_network.trainable_variables))

            self.log(loss_policy=policy_loss.numpy(),
                     gradients_norm_policy=[tf.norm(gradient) for gradient in policy_grads])

        # Value network optimization:
        for step, data_batch in enumerate(value_batches):
            with tf.GradientTape() as tape:
                value_loss = self.value_objective(batch=data_batch)

            value_grads = tape.gradient(value_loss, self.value_network.trainable_variables)
            self.value_optimizer.apply_gradients(zip(value_grads, self.value_network.trainable_variables))

            self.log(loss_value=value_loss.numpy(),
                     gradients_norm_value=[tf.norm(gradient) for gradient in value_grads])

        # Train exploration method
        self.exploration.train()

    def value_objective(self, batch):
        states, extrinsic_returns, intrinsic_returns = batch
        extrinsic_values, intrinsic_values = self.value_network(states, training=True)

        loss_e = losses.mean_squared_error(y_true=extrinsic_returns, y_pred=extrinsic_values)
        loss_i = losses.mean_squared_error(y_true=intrinsic_returns, y_pred=intrinsic_values)

        return tf.reduce_mean(loss_e * self.value_coeff[0] + loss_i * self.value_coeff[1])

    def get_advantages(self):
        adv_e = utils.gae(rewards=self.memory.extrinsic_rewards, values=self.memory.extrinsic_values,
                          gamma=self.gamma, lambda_=self.lambda_, normalize=True)

        # Don't normalize intrinsic advantages: because if seen states are no more novel,
        # the advantages will be close to 0, thus, not in the same range of the extrinsic ones.
        adv_i = utils.gae(rewards=self.memory.intrinsic_rewards, values=self.memory.intrinsic_values,
                          gamma=self.gamma, lambda_=self.lambda_, normalize=False)

        self.log(advantages_extrinsic=adv_e, advantages_intrinsic=adv_i)

        return adv_e * self.adv_weights[0] + adv_i * self.adv_weights[1]

    def get_returns(self):
        extrinsic_returns = utils.rewards_to_go(rewards=self.memory.extrinsic_rewards, discount=self.gamma,
                                                normalize=True)
        intrinsic_returns = utils.rewards_to_go(rewards=self.memory.intrinsic_rewards, discount=self.gamma,
                                                normalize=True)
        return extrinsic_returns, intrinsic_returns

    def learn(self, episodes: int, timesteps: int, batch_size: int, save_every: Union[bool, str, int] = False,
              render_every=0):
        if save_every is False:
            save_every = episodes + 1
        elif save_every is True:
            save_every = 1
        elif save_every == 'end':
            save_every = episodes

        with self.env as env:
            for episode in range(1, episodes + 1):
                self.memory = PPO2Memory(capacity=timesteps, states_shape=self.state_shape, num_actions=self.num_actions)
                state = env.reset()
                state = utils.to_tensor(state)
                episode_reward = 0.0
                t0 = time.time()
                render = episode % render_every == 0

                for t in range(1, timesteps + 1):
                    if render:
                        env.render()

                    # Compute action, log_prob, and value
                    policy = self.policy_network(state, training=False)
                    action = policy
                    log_prob = policy.log_prob(action)
                    value = self.value_network(state, training=False)

                    # Make action in the right range for the environment
                    converted_action = self.convert_action(action[0][0].numpy())

                    # Environment step
                    next_state, extrinsic_reward, done, _ = env.step(converted_action)
                    episode_reward += extrinsic_reward
                    intrinsic_reward = self.exploration.bonus(state)

                    self.log(actions=converted_action, rewards_extrinsic=extrinsic_reward,
                             rewards_intrinsic=intrinsic_reward)

                    self.memory.append(state, action, extrinsic_reward, intrinsic_reward, value, log_prob)
                    state = utils.to_tensor(next_state)

                    # check whether a termination (terminal state or end of a transition) is reached:
                    if done or (t == timesteps):
                        print(f'Episode {episode} terminated after {t} timesteps in {round((time.time() - t0), 4)}s ' +
                              f'with reward {episode_reward}.')
                        self.memory.end_trajectory(last_value=(0, 0) if done else self.value_network(state))
                        break

                self.update(batch_size)
                self.log(episode_rewards=episode_reward)
                self.write_summaries()

                if episode % save_every == 0:
                    self.save()

    def _value_network(self, units=32):
        """Dual-head Value Network"""
        inputs = Input(shape=self.state_shape, dtype=tf.float32)
        x = Dense(units, activation='tanh')(inputs)
        x = Dense(units, activation='relu')(x)
        x = Dense(units, activation='relu')(x)

        # Dual value head: intrinsic + extrinsic reward
        extrinsic_value = Dense(units=1, activation=None, name='extrinsic_head')(x)
        intrinsic_value = Dense(units=1, activation=None, name='intrinsic_head')(x)

        return Model(inputs, outputs=[extrinsic_value, intrinsic_value], name='value_network')


# TODO: must update
class PPO2Memory:
    def __init__(self, capacity: int, states_shape: tuple, num_actions: int):
        self.index = 0
        self.size = capacity

        # TODO: use 'tf' instead of 'np'
        self.states = np.zeros(shape=(capacity,) + states_shape, dtype=np.float32)
        self.extrinsic_rewards = np.zeros(shape=capacity + 1, dtype=np.float32)
        self.intrinsic_rewards = np.zeros(shape=capacity + 1, dtype=np.float32)
        self.extrinsic_values = np.zeros(shape=capacity + 1, dtype=np.float32)
        self.intrinsic_values = np.zeros(shape=capacity + 1, dtype=np.float32)
        self.actions = np.zeros(shape=(capacity, num_actions), dtype=np.float32)
        self.log_probabilities = np.zeros(shape=(capacity, 1), dtype=np.float32)

    def append(self, state, action, extrinsic_reward, intrinsic_reward, value, log_prob):
        assert self.index < self.size
        i = self.index
        value_e, value_i = value

        self.states[i] = tf.squeeze(state)
        self.actions[i] = tf.squeeze(action)
        self.extrinsic_rewards[i] = extrinsic_reward
        self.intrinsic_rewards[i] = intrinsic_reward
        self.extrinsic_values[i] = utils.tf_to_scalar_shape(value_e)
        self.intrinsic_values[i] = utils.tf_to_scalar_shape(value_i)
        self.log_probabilities[i] = log_prob
        self.index += 1

    def end_trajectory(self, last_value):
        """Terminates the current trajectory by adding the value of the terminal state"""
        value_e, value_i = last_value

        self.extrinsic_rewards[self.index] = value_e
        self.intrinsic_rewards[self.index] = value_i
        self.extrinsic_values[self.index] = value_e
        self.intrinsic_values[self.index] = value_i

        if self.index < self.size:
            # cut off the exceeding part
            self.states = self.states[:self.index]
            self.actions = self.actions[:self.index]
            self.extrinsic_rewards = self.extrinsic_rewards[:self.index + 1]
            self.intrinsic_rewards = self.intrinsic_rewards[:self.index + 1]
            self.extrinsic_values = self.extrinsic_values[:self.index + 1]
            self.intrinsic_values = self.intrinsic_values[:self.index + 1]
            self.log_probabilities = self.log_probabilities[:self.index]
