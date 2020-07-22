import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rl import utils
from rl.agents.agents import Agent

from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


class ReinforceAgent(Agent):
    # TODO: check and benchmark implementation
    def __init__(self, learning_rates=(3e-4, 3e-4), optimization_steps=(1, 2), name='reinforce-agent',
                 gamma=0.99, lambda_=0.95, target_kl=0.01, seed=None, weights_dir='weights', load=False):
        self.memory = None
        # self.distribution = None
        self.gamma = gamma
        self.lambda_ = lambda_
        self.target_kl = target_kl

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
        self.policy_optimizer = optimizers.Adam(learning_rate=learning_rates[0])
        self.value_optimizer = optimizers.Adam(learning_rate=learning_rates[1])
        self.optimization_steps = dict(policy=optimization_steps[0],
                                       value=optimization_steps[1])

        # Training Statistics
        self.stats = dict(policy_losses=[], value_losses=[], episode_rewards=[])

    def act(self, state):
        action = self.policy_network(state, training=False)
        return action.numpy()

    def predict(self, state):
        action = self.policy_network(state, training=False)
        log_prob = action.log_prob(action)
        value = self.value_network(state, training=False)
        return action, value, log_prob

    def update(self, batch_size: int):
        # Compute returns and advantages once:
        advantages = utils.gae(rewards=self.memory.rewards, values=self.memory.values,
                               gamma=self.gamma, lambda_=self.lambda_)

        returns = utils.rewards_to_go(rewards=self.memory.rewards, discount=self.gamma)

        # Prepare data: (states, returns) and (states, advantages)
        value_batches = utils.data_to_batches(tensors=(self.memory.states, returns[:-1]), batch_size=batch_size)
        policy_batches = utils.data_to_batches(tensors=(self.memory.states, advantages), batch_size=batch_size)

        # Policy network optimization:
        for step, (states_batch, advantages_batch) in enumerate(policy_batches):
            with tf.GradientTape() as tape:
                policy_loss = self.policy_gradient_objective(states=states_batch, advantages=advantages_batch)

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

    def policy_gradient_objective(self, states, advantages):
        actions = self.policy_network(states, training=True)

        return -actions.log_prob(actions) * advantages

    def learn(self, environment, episodes: int, timesteps: int, save=True):
        batch_size = timesteps
        print('batch_size:', batch_size)

        for episode in range(1, episodes + 1):
            self.memory = ReinforceMemory(capacity=timesteps)
            state = environment.reset()
            state = utils.to_tensor(state)
            episode_reward = 0.0

            for t in range(1, timesteps + 1):
                # environment.render()
                action, value, log_prob = self.predict(state)

                next_state, reward, done, _ = environment.step(action[0].numpy())
                episode_reward += reward

                self.memory.append(state, action, reward, value, log_prob)
                state = utils.to_tensor(next_state)

                # check whether a termination (terminal state or end of episode/transition) is reached:
                if done or (t == timesteps):
                    print(f'Episode {episode} terminated after {t} timesteps.')
                    self.memory.end_trajectory(last_value=0 if done else self.value_network(state)[0])
                    break

            self.update(batch_size)
            self.stats['episode_rewards'].append(episode_reward)

        if save:
            self.save()

    @staticmethod
    def categorical_policy_network(state_shape=(4,), num_actions=2):
        inputs = Input(shape=state_shape)
        x = Dense(24, activation='tanh')(inputs)
        x = Dense(48, activation='tanh')(x)
        logits = Dense(units=num_actions, activation='linear')(x)

        action = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfp.distributions.Categorical(logits=t))(logits)

        return Model(inputs, outputs=action)

    @staticmethod
    def _value_network(state_shape=(4,)):
        inputs = Input(shape=state_shape)
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


class ReinforceMemory:
    def __init__(self, capacity: int, states_shape=4, actions_shape=1):
        self.index = 0
        self.size = capacity

        # act
        self.states = np.zeros(shape=(capacity, states_shape), dtype=np.float32)
        self.actions = np.zeros(shape=(capacity, actions_shape), dtype=np.float32)  # TODO: maybe useless
        self.rewards = np.zeros(shape=capacity + 1, dtype=np.float32)

        # update
        self.values = np.zeros(shape=capacity + 1, dtype=np.float32)
        self.log_probabilities = np.zeros(shape=capacity, dtype=np.float32)

    def append(self, state, action, reward, value, log_prob):
        assert self.index < self.size
        i = self.index

        self.states[i] = tf.squeeze(state)
        self.actions[i] = action[0]
        self.rewards[i] = reward
        self.values[i] = tf.squeeze(value)
        self.log_probabilities[i] = log_prob[0]
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
