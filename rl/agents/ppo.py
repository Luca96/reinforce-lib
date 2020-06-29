import os
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
    # TODO: Polyak averaging
    def __init__(self, policy_lr=3e-4, value_lr=1e-4, optimization_steps=(10, 10), clip_ratio=0.2, load=False,
                 gamma=0.99, lambda_=0.95, target_kl=0.01, early_stop=False, seed=None, weights_dir='weights',
                 name='ppo-agent', use_log=False, use_summary=False):
        self.memory = None
        self.gamma = gamma
        self.lambda_ = lambda_
        self.target_kl = target_kl
        self.epsilon = clip_ratio
        self.early_stop = early_stop

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
            self.load()
        else:
            self.policy_network = self.categorical_policy_network()
            self.value_network = self._value_network()

        # Optimization
        self.policy_optimizer = optimizers.Adam(learning_rate=policy_lr)
        self.value_optimizer = optimizers.Adam(learning_rate=value_lr)
        self.optimization_steps = dict(policy=optimization_steps[0], value=optimization_steps[1])

        # Training Statistics
        self.stats = dict(policy_loss=[], value_loss=[], episode_rewards=[], ratio=[],
                          returns=[], values=[], advantages=[], prob=[], actions=[],
                          entropy=[], kl_divergence=[],  # policy_grads=[], value_grads=[]
                          )

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

        # returns = utils.rewards_to_go(rewards=self.memory.rewards, gamma=self.gamma)
        returns = utils.returns(rewards=self.memory.rewards, gamma=self.gamma)

        # Log
        self.log(returns=returns, advantages=advantages, values=self.memory.values)

        # Prepare data: (states, returns) and (states, advantages)
        value_batches = utils.data_to_batches(tensors=(self.memory.states, returns), batch_size=batch_size)
        policy_batches = utils.data_to_batches(tensors=(self.memory.states, advantages),
                                               batch_size=batch_size)

        # Initialize the old-policy (i.e. get the underlying tfp.Distribution object)
        old_policy = self.policy_network(utils.to_tensor(self.memory.states[0]), training=False)
        # old_policy = self.policy_network()

        # Policy network optimization:
        for step, (states_batch, advantages_batch) in enumerate(policy_batches):

            with tf.GradientTape() as tape:
                policy_loss, kl = self.ppo_clip_objective(old_policy, states=states_batch,
                                                          advantages=advantages_batch)

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

    def ppo_clip_objective(self, old_policy: tfp.distributions.Distribution, states, advantages):
        new_policy = self.policy_network(states, training=True)
        actions = new_policy

        new_log_prob = new_policy.log_prob(actions)
        old_log_prob = old_policy.log_prob(actions)
        kl_divergence = new_policy.kl_divergence(other=old_policy)

        # Compute the probability ratio between the current and old policy
        ratio = tf.math.exp(new_log_prob - old_log_prob)

        # Compute the clipped ratio times advantage (NOTE: this is the simplified PPO clip-objective):
        clipped_ratio = tf.where(advantages > 0, x=(1 + self.epsilon), y=(1 - self.epsilon))

        # Log stuff
        self.log(ratio=ratio, prob=np.exp(new_log_prob),
                 entropy=new_policy.entropy(), kl_divergence=kl_divergence)

        # Loss = min { ratio * A, clipped_ratio * A }
        # loss = -tf.minimum(ratio * advantages, clipped_ratio * advantages)
        loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
        return loss, tf.reduce_mean(kl_divergence).numpy()

    def learn(self, environment, episodes: int, timesteps: int, subsampling_fraction=0.25, save=True, render=True):
        batch_size = math.floor(timesteps * subsampling_fraction)
        print('batch_size:', batch_size)

        for episode in range(1, episodes + 1):
            self.memory = PPOMemory(capacity=timesteps)
            state = environment.reset()
            state = utils.to_tensor(state)
            episode_reward = 0.0
            t0 = time.time()

            for t in range(1, timesteps + 1):
                if render:
                    environment.render()

                # action = self.act(state)
                policy = self.policy_network(state, training=False)
                action = policy[0]
                log_prob = policy.log_prob(action)
                value = self.value_network(state, training=False)
                self.log(actions=action)

                next_state, reward, done, _ = environment.step(action)
                episode_reward += reward

                self.memory.append(state, reward, value, log_prob)
                state = utils.to_tensor(next_state)

                # check whether a termination (terminal state or end of a transition) is reached:
                if done or (t == timesteps):
                    print(f'Episode {episode} terminated after {t} timesteps in {round((time.time() - t0), 4)}ms')
                    self.memory.end_trajectory(last_value=0 if done else self.value_network(state)[0])
                    break

            self.update(batch_size)
            self.log(episode_rewards=episode_reward)

            if self.use_summary:
                self.write_summaries()

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
                    tf.summary.scalar(name=key, data=value, step=step + i)

                self.steps[key] += len(values)
                self.stats[key].clear()

    @staticmethod
    def categorical_policy_network(units=32, state_shape=(4,), num_actions=2):
        inputs = Input(shape=state_shape, dtype=tf.float32)
        x = Dense(units, activation='tanh')(inputs)
        x = Dense(units, activation='relu')(x)
        x = Dense(units, activation='relu')(x)
        logits = Dense(units=num_actions, activation=None)(x)

        action = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfp.distributions.Categorical(logits=t))(logits)

        return Model(inputs, outputs=action, name='policy')

    @staticmethod
    def _value_network(units=32, state_shape=(4,)):
        inputs = Input(shape=state_shape, dtype=tf.float32)
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
        print('loading...')
        self.policy_network = tf.keras.models.load_model(self.save_path['policy'])
        self.value_network = tf.keras.models.load_model(self.save_path['value'])

    def plot_statistics(self, colormap='Set3'):  # Pastel1, Set3, tab20b, tab20c
        """Colormaps: https://matplotlib.org/tutorials/colors/colormaps.html"""
        num_plots = len(self.stats.keys())
        cmap = plt.get_cmap(name=colormap)

        if math.sqrt(num_plots) == float(math.isqrt(num_plots)):
            rows = math.isqrt(num_plots)
            cols = rows
        else:
            rows = round(math.sqrt(num_plots))
            cols = math.ceil(math.sqrt(num_plots))

        for k, (key, value) in enumerate(self.stats.items()):
            plt.subplot(rows, cols, k + 1)
            plt.plot(value, color=cmap(k + 1))
            plt.title(key)

        plt.show()


class PPOMemory:
    def __init__(self, capacity: int, states_shape=4):
        self.index = 0
        self.size = capacity

        self.states = np.zeros(shape=(capacity, states_shape), dtype=np.float32)
        self.rewards = np.zeros(shape=capacity + 1, dtype=np.float32)
        self.values = np.zeros(shape=capacity + 1, dtype=np.float32)
        self.log_probabilities = np.zeros(shape=capacity, dtype=np.float32)

    def append(self, state, reward, value, log_prob):
        assert self.index < self.size
        i = self.index

        self.states[i] = tf.squeeze(state)
        self.rewards[i] = reward
        self.values[i] = tf.squeeze(value)
        self.log_probabilities[i] = tf.squeeze(log_prob)
        self.index += 1

    def end_trajectory(self, last_value):
        """Terminates the current trajectory by adding the value of the terminal state"""
        self.rewards[self.index] = last_value
        self.values[self.index] = last_value

        if self.index < self.size:
            # cut off the exceeding part
            self.states = self.states[:self.index]
            self.rewards = self.rewards[:self.index + 1]
            self.values = self.values[:self.index + 1]
            self.log_probabilities = self.log_probabilities[:self.index]

        # Normalize rewards to lower the value-network loss
        self.rewards = utils.np_normalize(self.rewards)
