import os
import math
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rl.agents import utils

from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


class PPOAgent:
    # TODO: Poliyak averaging
    def __init__(self, policy_lr=3e-4, value_lr=1e-4, optimization_steps=(10, 10), clip_ratio=0.2, load=False,
                 gamma=0.99, lambda_=0.95, target_kl=0.01, seed=None, weights_dir='weights', name='ppo-agent'):
        self.memory = None
        self.gamma = gamma
        self.lambda_ = lambda_
        self.target_kl = target_kl
        self.epsilon = clip_ratio
        self.seed = seed

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
        self.stats = dict(policy_losses=[], value_losses=[], episode_rewards=[], ratio=[],
                          returns=[], values=[], advantages=[], log_prob=[])

    def act(self, state):
        action = self.policy_network(state, training=False)
        return action[0].numpy()

    def update(self, batch_size: int):
        # Compute returns and advantages once:
        advantages = utils.generalized_advantage_estimation(rewards=self.memory.rewards, values=self.memory.values,
                                                            gamma=self.gamma, lambda_=self.lambda_,
                                                            normalize=False)

        returns = utils.rewards_to_go(rewards=self.memory.rewards, gamma=self.gamma)

        self.stats['returns'] = returns
        self.stats['advantages'] = advantages
        self.stats['values'] = self.memory.values

        # Prepare data: (states, returns) and (states, advantages)
        value_batches = utils.data_to_batches(tensors=(self.memory.states, returns[:-1]), batch_size=batch_size)
        policy_batches = utils.data_to_batches(tensors=(self.memory.states, advantages),
                                               batch_size=batch_size)

        # Initialize the old-policy (i.e. get the underlying tfp.Distribution object)
        old_policy = self.policy_network(utils.to_tensor(self.memory.states[0]), training=False)

        # Policy network optimization:
        for step, (states_batch, advantages_batch) in enumerate(policy_batches):
            with tf.GradientTape() as tape:
                policy_loss = self.ppo_clip_objective(old_policy, states=states_batch,
                                                      advantages=advantages_batch)

            policy_grads = tape.gradient(policy_loss, self.policy_network.trainable_weights)
            self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy_network.trainable_weights))

            self.stats['policy_losses'].append(tf.reduce_mean(policy_loss).numpy())

        # Value network optimization:
        for step, (states_batch, returns_batch) in enumerate(value_batches):
            with tf.GradientTape() as tape:
                value_loss = self.value_objective(states=states_batch, returns=returns_batch)

            value_grads = tape.gradient(value_loss, self.value_network.trainable_weights)
            self.value_optimizer.apply_gradients(zip(value_grads, self.value_network.trainable_weights))

            self.stats['value_losses'].append(tf.reduce_mean(value_loss).numpy())

    def value_objective(self, states, returns):
        values = self.value_network(states, training=True)

        return losses.mean_squared_error(y_true=returns, y_pred=values)

    def ppo_clip_objective(self, old_policy: tfp.distributions.Distribution, states, advantages):
        new_policy = self.policy_network(states, training=True)
        actions = new_policy

        self.stats['log_prob'].extend(np.exp(new_policy.log_prob(actions)))

        # Compute the probability ratio between the current and old policy
        ratio = tf.math.exp(new_policy.log_prob(actions) - old_policy.log_prob(actions))
        self.stats['ratio'].append(tf.reduce_mean(ratio))

        # Compute the clipped ratio times advantage (NOTE: this is the simplified PPO clip-objective):
        clipped_ratio = tf.where(advantages > 0, x=(1 + self.epsilon), y=(1 - self.epsilon))

        # Loss = min { ratio * A, clipped_ratio * A }
        return -tf.minimum(ratio * advantages, clipped_ratio * advantages)

    def learn(self, environment, episodes: int, timesteps: int, subsampling_fraction=0.25, save=True, render=True):
        batch_size = math.floor(timesteps * subsampling_fraction)
        print('batch_size:', batch_size)
        best_reward = -math.inf

        for episode in range(1, episodes + 1):
            self.memory = PPOMemory(capacity=timesteps)
            state = environment.reset()
            state = utils.to_tensor(state)
            episode_reward = 0.0

            for t in range(1, timesteps + 1):
                if render:
                    environment.render()

                action = self.act(state)
                value = self.value_network(state, training=False)

                next_state, reward, done, _ = environment.step(action)
                episode_reward += reward

                self.memory.append(state, reward, value)
                state = utils.to_tensor(next_state)

                # check whether a termination (terminal state or end of a transition) is reached:
                if done or (t == timesteps):
                    print(f'Episode {episode} terminated after {t} timesteps.')
                    self.memory.end_trajectory(last_value=0 if done else self.value_network(state)[0])
                    break

            self.update(batch_size)
            self.stats['episode_rewards'].append(episode_reward)

            if episode_reward >= best_reward:
                best_reward = episode_reward

                if save:
                    self.save()

    def evaluate(self, episodes: int, timesteps: int):
        pass

    @staticmethod
    def categorical_policy_network(state_shape=(4,), num_actions=2):
        inputs = Input(shape=state_shape, dtype=tf.float32)
        x = Dense(24, activation='tanh')(inputs)
        x = Dense(48, activation='tanh')(x)
        logits = Dense(units=num_actions, activation='linear')(x)

        action = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfp.distributions.Categorical(logits=t))(logits)

        return Model(inputs, outputs=action)

    @staticmethod
    def _value_network(state_shape=(4,)):
        inputs = Input(shape=state_shape, dtype=tf.float32)
        x = Dense(24, activation='tanh')(inputs)
        x = Dense(48, activation='tanh')(x)
        output = Dense(units=1, activation='linear')(x)
        return Model(inputs, output)

    def save(self):
        print('saving...')
        self.policy_network.save(self.save_path['policy'])
        self.value_network.save(self.save_path['value'])

    def load(self):
        print('loading...')
        self.policy_network = tf.keras.models.load_model(self.save_path['policy'])
        self.value_network = tf.keras.models.load_model(self.save_path['value'])


class PPOMemory:
    def __init__(self, capacity: int, states_shape=4):
        self.index = 0
        self.size = capacity

        self.states = np.zeros(shape=(capacity, states_shape), dtype=np.float32)
        self.rewards = np.zeros(shape=capacity + 1, dtype=np.float32)
        self.values = np.zeros(shape=capacity + 1, dtype=np.float32)

    def append(self, state, reward, value):
        assert self.index < self.size
        i = self.index

        self.states[i] = tf.squeeze(state)
        self.rewards[i] = reward
        self.values[i] = tf.squeeze(value)
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
