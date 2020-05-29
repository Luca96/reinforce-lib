import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

from rl.memories import Recent, Replay


class Agent(object):
    def __init__(self, state_shape: tuple, action_shape: tuple, batch_size: int,
                 gamma=0.99, lambda_=0.95):
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

    def reset(self):
        pass

    def act(self, state):
        action = self.policy_network(state)
        self.current['state'] = state
        self.current['action'] = action
        return action

    def observe(self, next_state, reward, terminal) -> bool:
        # transition = [state, reward, terminal]
        # transition = dict(state=state, reward=reward, terminal=terminal)
        transition = dict(state=self.current['state'], action=self.current['action'],
                          reward=reward, next_state=next_state, terminal=terminal)
        self.memory.append(transition)

        if terminal:
            self.update()

        return terminal  # i.e. whether updated or not

    def update(self):
        raise NotImplementedError

    # def generalized_advantage_estimation(self, batch: list) -> float:
    #     rewards = [transition['reward'] for transition in batch]
    #     states = [transition['state'] for transition in batch]
    #     advantage = 0
    #     lam = 1
    #     values = self.value_network(states)
    #
    #     for k in range(len(batch)):
    #         adv = -values[0]
    #         gamma = 1
    #
    #         for i in range(k + 1):
    #             adv += gamma * rewards[i]
    #             gamma *= gamma
    #
    #         advantage += lam * (adv + gamma * values[k])
    #         lam *= self.lam
    #
    #     return (1 - self.lam) * advantage

    # def generalized_advantage_estimation(self, batch: list) -> float:
    #     rewards = [transition['reward'] for transition in batch]
    #     states = [transition['state'] for transition in batch]
    #     values = self.value_network(states)
    #     k = len(batch)
    #
    #     advantage = -k * values[0]
    #     gamma = self.gamma
    #
    #     for i in range(0, k - 1):
    #         advantage += gamma * (k - i) * rewards[i]  # sum discounted reward: gamma^i r_i
    #         gamma *= self.gamma
    #         advantage += gamma * values[i + 1]  # sum next value function: gamma^(i+1) V(s_i+1)
    #
    #     return (1 - self.lam) * advantage

    def gae(self, batch) -> float:
        """Generalized Advantage Estimation"""
        rewards = [transition['reward'] for transition in batch]
        states = np.array([transition['state'] for transition in batch])
        values = [v[0].numpy() for v in self.value_network(states.reshape(-1, 1))]

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

    def _build_value_network(self):
        inputs = Input(shape=self.state_shape)
        x = Dense(32, activation='relu')(inputs)
        x = Dense(32, activation='relu')(x)
        output = Dense(units=1, activation=None)(x)
        return Model(inputs, output)

# Incremental Monte-Carlo Learning: v(s_t) = v(s_t) + alpha * (return_t - v(s_t))
# TD-learning(0): v(s_t) = v(s_t) + alpha * (r_t + gamma * v(s_t+1) - v(s_t))
# TD(0) + n-step return: v(s_t) = v(s_t) + alpha * (G_t^n - v(s_t))
#  dove: G_t^n = (r_t+1 + ... + r_t+n) + gamma^n v(s_t+n)


class ReinforceAgent(Agent):

    def update(self):
        # TODO: batch_size should be 'horizon'
        trajectory = self.memory.retrieve(amount=self.batch_size)
        self.memory.clear()

        # TODO: here we consider only one trajectory, should be 'batch_size' instead!
        discounted_sum = self._discounted_sum_of_rewards(batch=trajectory)

        # update weights: w + alpha * g
        # history = self.value_network.fit(x=)

    def _discounted_sum_of_rewards(self, batch) -> float:
        discounted_sum = 0.0

        for transition in batch:
            discounted_sum += self.gamma * transition['reward']

        return discounted_sum
