from typing import Tuple, Any, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# from tensorflow import keras
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from rl.memories import Recent, Replay
# eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0


class Agent(object):
    # TODO: use specifications for state and action spaces
    def __init__(self, state_shape: tuple, action_shape: tuple, batch_size: int,
                 gamma=0.99, lambda_=0.95, policy_lr=3e-4, value_lr=3e-4):
        self.batch_size = batch_size
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.state_batch_shape = (batch_size,) + state_shape
        self.action_batch_shape = (batch_size,) + action_shape

        # Reward estimation (GAE)
        self.gamma = gamma  # discount factor
        self.lambda_ = lambda_
        self.current = dict(state=None, action=None)
        # self.horizon  # bootstrap reward estimation

        # Memory (contains transitions: {s, a, r, d})
        self.memory = Recent(capacity=batch_size)

        self.policy_network = self._build_policy_network()
        self.value_network = self._build_value_network()
        self.distribution = None  # pi/policy
        self.policy_optimizer = optimizers.Adam(learning_rate=policy_lr)
        self.value_optimizer = optimizers.Adam(learning_rate=value_lr)

        self.timestep = 0
        self.episodes = 0
        self.statistics = dict(histories=[], losses=[])

    def close(self):
        self.memory.clear()
        self.distribution = None
        self.current = dict(state=None, action=None)
        self.timestep = 0
        self.episodes = 0
        self.statistics = dict(histories=[], losses=[])

    def act(self, state):
        action = self.policy_network(state, training=True)[0]
        self.timestep += 1
        self.current['state'] = state
        self.current['action'] = action
        return action.numpy()

    def evaluate(self):
        raise NotImplementedError

    def observe(self, next_state, reward, terminal) -> bool:
        # transition = [state, reward, terminal]
        # transition = dict(state=state, reward=reward, terminal=terminal)
        transition = dict(state=self.current['state'],
                          action=self.current['action'],
                          reward=reward, next_state=next_state, terminal=terminal)
        self.memory.append(transition)

        if terminal:
            self.episodes += 1

        if (self.timestep % self.batch_size == 0) or terminal:
            print(f'Update at t={self.timestep} and e={self.episodes}')
            self.update()
            return True

        return False

    def learn(self, environment, num_episodes: int, max_timesteps: int, **kwargs):
        raise NotImplementedError

    def gae(self, batch) -> float:
        """Generalized Advantage Estimation"""
        rewards = [transition['reward'] for transition in batch]
        states = np.array([transition['state'][0] for transition in batch])
        values = [v[0].numpy() for v in self.value_network(states)]

        def tf_target(t: int):
            return rewards[t] + self.gamma * values[t + 1] - values[t]

        advantage = 0.0
        gamma_lambda = 1

        for i in range(len(batch) - 1):
            advantage += tf_target(i) * gamma_lambda
            gamma_lambda *= self.gamma * self.lambda_

        return advantage

    def _build_policy_network(self):
        # inputs = Input(shape=self.obs_shape, batch_size=self.batch_size)
        inputs = Input(shape=self.state_shape)
        x = Dense(32, activation='relu')(inputs)
        x = Dense(32, activation='relu')(x)
        output = Dense(units=np.prod(self.action_shape), activation=None)(x)
        return Model(inputs, output)

    def _build_value_network(self, learning_rate=0.001):
        # build model
        inputs = Input(shape=self.state_shape)
        x = Dense(32, activation='relu')(inputs)
        x = Dense(32, activation='relu')(x)
        output = Dense(units=1, activation=None)(x)
        return Model(inputs, output)


class CategoricalReinforceAgent(Agent):

    def learn(self, environment, num_episodes: int, max_timesteps: int, **kwargs):
        statistics = dict(rewards=[], policy_losses=[], value_losses=[])

        for episode in range(1, num_episodes + 1):
            state = environment.reset()
            state = self._to_tensor(state)
            episode_reward = 0.0

            with tf.GradientTape(persistent=True) as tape:
                for t in range(1, max_timesteps + 1):
                    environment.render()
                    action = self.act(state=state)

                    state, reward, done, _ = environment.step(action)
                    state = self._to_tensor(state)
                    episode_reward += reward

                    self.observe(next_state=state, reward=reward, terminal=done)

                    if done:
                        statistics['rewards'].append(episode_reward)
                        print(f"Episode {episode} terminated at timestep {t} with reward {round(episode_reward, 2)}.")
                        break

                batch = self.memory.retrieve(amount=self.memory.size)
                value_loss = self.value_loss(batch)
                policy_loss = self.policy_gradient_loss(batch)

                # update training statistics
                statistics['value_losses'].append(np.mean(value_loss))
                statistics['policy_losses'].append(np.mean(policy_loss))

            self.memory.clear()

            # update step on value network
            value_gradients = tape.gradient(value_loss, self.value_network.trainable_weights)
            self.value_optimizer.apply_gradients(zip(value_gradients, self.value_network.trainable_weights))

            # update step on policy network
            policy_gradients = tape.gradient(policy_loss, self.policy_network.trainable_weights)
            self.policy_optimizer.apply_gradients(zip(policy_gradients, self.policy_network.trainable_weights))

        environment.close()
        self.close()
        return statistics

    def observe(self, next_state, reward, terminal):
        transition = dict(state=self.current['state'],
                          action=self.current['action'],
                          reward=reward, next_state=next_state, terminal=terminal)
        self.memory.append(transition)

    def value_loss(self, batch):
        states = [x['state'] for x in batch]
        rewards = [x['reward'] for x in batch]
        next_states = [x['next_state'] for x in batch]

        # predict and compute values
        estimated_values = self.value_network(np.array(states).squeeze(), training=True)
        target_values = self._target_values(rewards, next_states)

        # try: losses.squared_hinge
        return losses.mean_squared_error(y_true=target_values, y_pred=estimated_values)

    def policy_gradient_loss(self, batch):
        actions = [transition['action'] for transition in batch]
        advantages = self.gae(batch)
        log_prob = self.distribution.log_prob(value=actions)
        return -log_prob * advantages

    @staticmethod
    def _to_tensor(x, expand_axis=0):
        x = tf.convert_to_tensor(x)
        x = tf.expand_dims(x, axis=expand_axis)
        return x

    def get_distribution(self, t):
        self.distribution = tfp.distributions.Categorical(logits=t)
        return self.distribution

    def _build_policy_network(self, num_actions=2):
        inputs = Input(shape=self.state_shape, batch_size=1)
        x = Dense(24, activation='tanh')(inputs)
        x = Dense(48, activation='tanh')(x)
        logits = Dense(units=num_actions, activation='linear')(x)

        categorical = tfp.layers.DistributionLambda(
            make_distribution_fn=self.get_distribution)(logits)

        return Model(inputs, categorical)

    def _target_values(self, rewards, next_states) -> list:
        next_state_values = self.value_network(np.array(next_states).squeeze(), training=True)
        values = []

        discounted_sum = 0.0
        discount_factor = 1

        for i, reward in enumerate(rewards):
            discounted_sum += discount_factor * reward
            discount_factor *= self.gamma

            value = discounted_sum + discount_factor * next_state_values[i]
            values.append(value)

        return values
