import os
import gym
import time
import math
import random
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from datetime import datetime
from rl.agents import utils

from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


class PPOAgent:
    # TODO: same 'optimization steps' for both policy and value functions?
    # TODO: use Beta distribution for bounded continuous actions
    # TODO: 'value_loss' a parameter that selects the loss (either 'mse' or 'huber') for the value network
    # TODO: try 'mixture' of Beta/Gaussian distribution
    def __init__(self, environment: gym.Env, policy_lr=3e-4, value_lr=1e-4, optimization_steps=(10, 10), clip_ratio=0.2,
                 load=False,
                 gamma=0.99, lambda_=0.95, target_kl=0.01, entropy_regularization=0.0, early_stop=False, seed=None,
                 weights_dir='weights', name='ppo-agent', use_log=False, use_summary=False):
        self.memory = None
        self.gamma = gamma
        self.lambda_ = lambda_
        self.target_kl = target_kl
        self.entropy_strength = entropy_regularization
        self.epsilon = clip_ratio
        self.early_stop = early_stop
        self.env = environment

        # State/action space
        if isinstance(self.env.observation_space, gym.spaces.Box):
            self.state_shape = self.env.observation_space.shape
        else:
            self.state_shape = (self.env.observation_space.n,)

        if isinstance(self.env.action_space, gym.spaces.Box):
            self.num_actions = self.env.action_space.shape[0]

            if self.env.action_space.is_bounded():
                self.distribution_type = 'beta'
                self.action_range = self.env.action_space.high - self.env.action_space.low
                self.convert_action = lambda a: a * self.action_range + self.env.action_space.low

                assert self.action_range == abs(self.env.action_space.low - self.env.action_space.high)
            else:
                self.distribution_type = 'gaussian'
                self.convert_action = lambda a: a
        else:
            self.num_actions = 1
            self.distribution_type = 'categorical'
            self.convert_action = lambda a: a

        print('state_shape:', self.state_shape)
        print('action_shape:', self.num_actions)
        print('distribution:', self.distribution_type)

        # Logging
        self.use_log = use_log
        self.use_summary = use_summary

        if self.use_summary:
            self.summary_dir = os.path.join('logs', name, datetime.now().strftime("%Y%m%d-%H%M%S"))
            self.tf_summary_writer = tf.summary.create_file_writer(self.summary_dir, max_queue=5)

        # # Set random seed:
        # if seed is not None:
        #     tf.random.set_seed(seed)
        #     np.random.seed(seed)
        #     random.seed(seed)
        #     print(f'Random seed {seed}.')

        # Saving stuff:
        self.base_path = os.path.join(weights_dir, name)
        self.save_path = dict(policy=os.path.join(self.base_path, 'policy_net'),
                              value=os.path.join(self.base_path, 'value_net'))
        # Networks
        if load:
            # TODO: when loading the model save: iteration number (for logs), dynamic parameters (e.g. learning rate) ..
            self.load()
        else:
            self.policy_network = self._policy_network()
            self.value_network = self._value_network()

        # Optimization
        self.policy_optimizer = optimizers.Adam(learning_rate=policy_lr)
        self.value_optimizer = optimizers.Adam(learning_rate=value_lr)
        self.optimization_steps = dict(policy=optimization_steps[0], value=optimization_steps[1])

        # Training Statistics
        self.stats = dict(policy_loss=[], value_loss=[], episode_rewards=[], ratio=[],
                          returns=[], values=[], advantages=[], prob=[], actions=[],
                          entropy=[], kl_divergence=[])

        self.steps = dict(policy_loss=0, value_loss=0, episode_rewards=0, ratio=0,
                          returns=0, values=0, advantages=0, prob=0, actions=0,
                          entropy=0, kl_divergence=0)

    def act(self, state):
        action = self.policy_network(state, training=False)
        return action[0].numpy()

    def update(self, batch_size: int):
        # Compute returns and advantages once:
        advantages = utils.gae(rewards=self.memory.rewards, values=self.memory.values,
                               gamma=self.gamma, lambda_=self.lambda_,
                               normalize=True)

        returns = utils.rewards_to_go(rewards=self.memory.rewards, gamma=self.gamma,
                                      normalize=True)

        # Log
        self.log(returns=returns, advantages=advantages, values=tf.squeeze(self.memory.values))

        # Prepare data: (states, returns) and (states, advantages)
        value_batches = utils.data_to_batches(tensors=(self.memory.states, returns), batch_size=batch_size)
        policy_batches = utils.data_to_batches(tensors=(self.memory.states, advantages,
                                                        self.memory.actions, self.memory.log_probabilities),
                                               batch_size=batch_size)

        # Policy network optimization:
        for step, batch in enumerate(policy_batches):

            with tf.GradientTape() as tape:
                policy_loss, kl = self.ppo_clip_objective(batch)

            policy_grads = tape.gradient(policy_loss, self.policy_network.trainable_variables)
            self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy_network.trainable_variables))

            self.log(policy_loss=policy_loss.numpy())

            # Stop early if target_kl is reached:
            if self.early_stop and (kl > 1.5 * self.target_kl):
                print(f'early stop at step {step}.')
                break

        # Value network optimization:
        for step, (states_batch, returns_batch) in enumerate(value_batches):
            with tf.GradientTape() as tape:
                value_loss = self.value_objective(states=states_batch, returns=returns_batch)

            value_grads = tape.gradient(value_loss, self.value_network.trainable_weights)
            self.value_optimizer.apply_gradients(zip(value_grads, self.value_network.trainable_weights))

            self.log(value_loss=value_loss.numpy())

    def value_objective(self, states, returns):
        values = self.value_network(states, training=True)

        return tf.reduce_mean(losses.mean_squared_error(y_true=returns, y_pred=values))

    def ppo_clip_objective(self, batch):
        states, advantages, actions, old_log_probabilities = batch
        new_policy: tfp.distributions.Distribution = self.policy_network(states, training=True)

        new_log_prob = new_policy.log_prob(actions)

        # TODO: find a better way to compute the KL-divergence
        # kl_divergence = new_policy.kl_divergence(other=old_policy)
        kl_divergence = np.mean(new_log_prob - old_log_probabilities)

        # Entropy
        entropy = new_policy.entropy()
        entropy_term = self.entropy_strength * entropy

        # Compute the probability ratio between the current and old policy
        ratio = tf.math.exp(new_log_prob - old_log_probabilities)

        # Compute the clipped ratio times advantage (NOTE: this is the simplified PPO clip-objective):
        clipped_ratio = tf.where(advantages > 0, x=(1 + self.epsilon), y=(1 - self.epsilon))

        # Log stuff
        self.log(ratio=ratio,
                 prob=tf.exp(new_log_prob),
                 entropy=entropy,
                 kl_divergence=kl_divergence)

        # Loss = min { ratio * A, clipped_ratio * A } + entropy_term
        loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages) + entropy_term)
        return loss, kl_divergence

    def learn(self, episodes: int, timesteps: int, batch_size: int, save_every=-1, render_every=0):
        """
        :param episodes:
        :param timesteps:
        :param batch_size:
        :param save_every: '-1' means never save, '0' means always. Saves when episode % save_every == 0
        :param render_every:
        :return:
        """
        env = self.env

        for episode in range(1, episodes + 1):
            self.memory = PPOMemory(capacity=timesteps, states_shape=self.state_shape, num_actions=self.num_actions)
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
                action = policy[0][0].numpy()
                log_prob = policy.log_prob(action)
                value = self.value_network(state, training=False)

                # Make action in the right range for the environment
                converted_action = self.convert_action(action)
                self.log(actions=converted_action)

                next_state, reward, done, _ = env.step(converted_action)
                episode_reward += reward

                self.memory.append(state, action, reward, value, log_prob)
                state = utils.to_tensor(next_state)

                # check whether a termination (terminal state or end of a transition) is reached:
                if done or (t == timesteps):
                    print(f'Episode {episode} terminated after {t} timesteps in {round((time.time() - t0), 4)}s ' +
                          f'with reward {episode_reward}.')
                    self.memory.end_trajectory(last_value=0 if done else self.value_network(state)[0])
                    break

            self.update(batch_size)
            self.log(episode_rewards=episode_reward)

            if self.use_summary:
                self.write_summaries()

            if episode % save_every == 0:
                self.save()

    def evaluate(self, episodes: int, timesteps: int):
        pass

    def log(self, **kwargs):
        if self.use_log:
            for key, value in kwargs.items():
                if hasattr(value, '__iter__'):
                    self.stats[key].extend(value)
                else:
                    self.stats[key].append(value)

    def write_summaries(self):
        with self.tf_summary_writer.as_default():
            for key, values in self.stats.items():
                step = self.steps[key]

                for i, value in enumerate(values):
                    tf.summary.scalar(name=key, data=np.squeeze(value), step=step + i)

                self.steps[key] += len(values)
                self.stats[key].clear()

    def _policy_network(self, units=32):
        inputs = Input(shape=self.state_shape, dtype=tf.float32)
        x = Dense(units, activation='tanh')(inputs)
        x = Dense(units, activation='relu')(x)
        x = Dense(units, activation='relu')(x)
        action = self.get_distribution_layer(layer=x)

        return Model(inputs, outputs=action, name='policy')

    def get_distribution_layer(self, layer: Layer) -> tfp.layers.DistributionLambda:

        if self.distribution_type == 'categorical':
            # Categorical
            logits = Dense(units=self.env.action_space.n, activation=None)(layer)

            return tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Categorical(logits=t),
                convert_to_tensor_fn=lambda s: s.sample(self.num_actions))(logits)

        elif self.distribution_type == 'beta':
            # Beta
            # for activations choice see chapter 4 of http://proceedings.mlr.press/v70/chou17a/chou17a.pdf
            alpha = Dense(units=self.num_actions, activation='softplus')(layer)
            alpha = Add()([alpha, tf.ones_like(alpha)])

            beta = Dense(units=self.num_actions, activation='softplus')(layer)
            beta = Add()([beta, tf.ones_like(beta)])

            return tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Beta(t[0], t[1]),
                convert_to_tensor_fn=lambda s: s.sample(self.num_actions))([alpha, beta])

        # Gaussian (Normal)
        # for activations choice see chapter 4 of http://proceedings.mlr.press/v70/chou17a/chou17a.pdf
        mu = Dense(units=self.num_actions, activation='linear')(layer)
        sigma = Dense(units=self.num_actions, activation='softplus')(layer)

        return tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfp.distributions.Normal(loc=t[0], scale=t[..., 1]),
            convert_to_tensor_fn=lambda s: s.sample(self.num_actions))([mu, sigma])

    def _value_network(self, units=32):
        inputs = Input(shape=self.state_shape, dtype=tf.float32)
        x = Dense(units, activation='tanh')(inputs)
        x = Dense(units, activation='relu')(x)
        x = Dense(units, activation='relu')(x)
        output = Dense(units=1, activation=None)(x)
        return Model(inputs, output, name='value')

    def save(self):
        print('saving...')
        self.policy_network.save(self.save_path['policy'])
        self.value_network.save(self.save_path['value'])

    def load(self):
        # TODO: loading bugged with distribution objects!
        print('loading...')
        self.policy_network = tf.keras.models.load_model(self.save_path['policy'], compile=False)
        self.value_network = tf.keras.models.load_model(self.save_path['value'], compile=False)

    def plot_statistics(self, colormap='Set3'):  # Pastel1, Set3, tab20b, tab20c
        """Colormaps: https://matplotlib.org/tutorials/colors/colormaps.html"""
        num_plots = len(self.stats.keys())
        cmap = plt.get_cmap(name=colormap)
        rows = round(math.sqrt(num_plots))
        cols = math.ceil(math.sqrt(num_plots))

        for k, (key, value) in enumerate(self.stats.items()):
            plt.subplot(rows, cols, k + 1)
            plt.plot(value, color=cmap(k + 1))
            plt.title(key)

        plt.show()


class PPOMemory:
    def __init__(self, capacity: int, states_shape: tuple, num_actions: int):
        self.index = 0
        self.size = capacity

        self.states = np.zeros(shape=(capacity,) + states_shape, dtype=np.float32)
        self.rewards = np.zeros(shape=capacity + 1, dtype=np.float32)
        self.values = np.zeros(shape=capacity + 1, dtype=np.float32)
        self.actions = np.zeros(shape=(capacity, num_actions), dtype=np.float32)
        self.log_probabilities = np.zeros(shape=(capacity, 1), dtype=np.float32)

    def append(self, state, action, reward, value, log_prob):
        assert self.index < self.size
        i = self.index

        self.states[i] = tf.squeeze(state)
        self.actions[i] = action
        self.rewards[i] = reward
        self.values[i] = utils.tf_to_scalar_shape(value)
        self.log_probabilities[i] = utils.tf_to_scalar_shape(log_prob)
        self.index += 1

    def end_trajectory(self, last_value):
        """Terminates the current trajectory by adding the value of the terminal state"""
        self.rewards[self.index] = last_value
        self.values[self.index] = last_value

        if self.index < self.size:
            # cut off the exceeding part
            self.states = self.states[:self.index]
            self.actions = self.actions[:self.index]
            self.rewards = self.rewards[:self.index + 1]
            self.values = self.values[:self.index + 1]
            self.log_probabilities = self.log_probabilities[:self.index]
