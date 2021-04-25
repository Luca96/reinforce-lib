"""Synchronous Advantage Actor-Critic (A2C)"""

import os
import tensorflow as tf
import numpy as np

from rl import utils
from rl.parameters import DynamicParameter

from rl.v2.agents import ParallelAgent
from rl.v2.memories import TransitionSpec
from rl.v2.memories.episodic import EpisodicMemory
from rl.v2.networks import Network, ValueNetwork, PolicyNetwork

from typing import List, Tuple, Union, Dict


# TODO: check update method
class A2C(ParallelAgent):
    """Sequential (single-process) implementation of A2C"""

    def __init__(self, env, name='a2c-agent', parallel_actors=16, horizon=5, entropy=0.01, load=False, gamma=0.99,
                 optimizer='rmsprop', lambda_=1.0, normalize_advantages: Union[None, str] = 'sign', actor: dict = None,
                 critic: dict = None, advantage_scale: utils.DynamicType = 1.0, actor_lr: utils.DynamicType = 7e-4,
                 clip_norm: Tuple[utils.DynamicType] = (1.0, 1.0), critic_lr: utils.DynamicType = 7e-4, **kwargs):
        assert horizon >= 1
        self.horizon = int(horizon)

        super().__init__(env, batch_size=self.horizon, num_actors=parallel_actors, gamma=gamma, name=name, **kwargs)

        self.lambda_ = tf.constant(lambda_, dtype=tf.float32)
        self.entropy_strength = DynamicParameter.create(value=entropy)
        self.adv_scale = DynamicParameter.create(value=advantage_scale)
        self.adv_normalization_fn = utils.get_normalization_fn(arg=normalize_advantages)

        self.actor_lr = DynamicParameter.create(value=actor_lr)
        self.critic_lr = DynamicParameter.create(value=critic_lr)

        # shared networks (and optimizer)
        self.actor = Network.create(agent=self, log_prefix='actor', **(actor or {}), base_class='A2C-ActorNetwork')
        self.critic = Network.create(agent=self, log_prefix='critic', **(critic or {}), base_class='A2C-CriticNetwork')

        # self.actor = ActorNetwork(agent=self, log_prefix='actor', **(actor or {}))
        # self.critic = CriticNetwork(agent=self, log_prefix='critic', **(critic or {}))

        if isinstance(optimizer, dict):
            opt_args = optimizer
            optimizer = opt_args.pop('name', 'rmsprop')
        else:
            opt_args = {}

        self.actor.compile(optimizer, clip_norm=clip_norm[0], learning_rate=self.actor_lr, **opt_args)
        self.critic.compile(optimizer, clip_norm=clip_norm[1], learning_rate=self.critic_lr, **opt_args)

        self.weights_path = dict(policy=os.path.join(self.base_path, 'actor'),
                                 value=os.path.join(self.base_path, 'critic'))

        if load:
            self.load()

    @property
    def transition_spec(self) -> TransitionSpec:
        state_spec = {k: (self.horizon,) + shape for k, shape in self.state_spec.items()}

        return TransitionSpec(state=state_spec, action=(self.horizon, self.num_actions), next_state=False,
                              terminal=False, reward=(self.horizon, 1), other=dict(value=(self.horizon, 1)))

    @property
    def memory(self) -> 'ParallelGAEMemory':
        if self._memory is None:
            self._memory = ParallelGAEMemory(self.transition_spec, agent=self, shape=self.num_actors)

        return self._memory

    def act(self, states) -> Tuple[tf.Tensor, dict, dict]:
        actions, _, means, std = self.actor(states, training=False)
        values = self.critic(states, training=False)

        other = dict(value=values)
        debug = dict()

        if self.distribution_type != 'categorical':
            for i, (mu, sigma) in enumerate(zip(means, std)):
                debug[f'distribution_mean_{i}'] = mu
                debug[f'distribution_std_{i}'] = sigma

        return actions, other, debug

    @staticmethod
    def average_gradients(gradients_list) -> list:
        n = 1.0 / len(gradients_list)

        gradients = gradients_list[0]

        for i in range(1, len(gradients_list)):
            grads = gradients_list[i]

            for j, g in enumerate(grads):
                gradients[j] += g

        return [g * n for g in gradients]

    def update(self):
        batches = self.memory.get_data()

        all_batches = {k: v for k, v in batches[0].items()}
        for i in range(1, len(batches)):
            for k, v in batches[i].items():
                all_batches[k] = tf.concat([v, all_batches[k]], axis=0)

        actor_grads = self.actor.train_step(all_batches)
        critic_grads = self.critic.train_step(all_batches)

        # update weights
        self.actor.update(gradients=actor_grads)
        self.critic.update(gradients=critic_grads)

    def on_transition(self, transition: Dict[str, list], timestep: int, episode: int, exploration=False):
        super().on_transition(transition, timestep, episode, exploration)

        if any(transition['terminal']) or (timestep % self.horizon == 0) or (timestep == self.max_timesteps):
            terminal_states = self.preprocess(transition['next_state'])

            values = self.critic(terminal_states, training=False)
            values = values * utils.to_float(tf.logical_not(transition['terminal']))

            debug = self.memory.end_trajectory(last_values=values)
            self.log(average=True, **debug)

            if not exploration:
                self.update()
                self.memory.clear()

    def load_weights(self):
        self.actor.load_weights(filepath=self.weights_path['actor'], by_name=False)
        self.critic.load_weights(filepath=self.weights_path['critic'], by_name=False)

    def save_weights(self):
        self.actor.save_weights(filepath=self.weights_path['actor'])
        self.critic.save_weights(filepath=self.weights_path['critic'])

    def summary(self):
        self.actor.summary()
        self.critic.summary()


@Network.register(name='A2C-ActorNetwork')
class ActorNetwork(PolicyNetwork):

    def train_step(self, batch: dict):
        if isinstance(batch, tuple):
            batch = batch[0]

        debug, grads = self.train_on_batch(batch)
        self.agent.log(average=True, **({f'{self.prefix}_{k}': v for k, v in debug.items()}))

        return grads

    def objective(self, batch, reduction=tf.reduce_sum) -> tuple:
        return super().objective(batch, reduction=reduction)

    @tf.function
    def train_on_batch(self, batch):
        with tf.GradientTape() as tape:
            loss, debug = self.objective(batch)

        gradients = tape.gradient(loss, self.trainable_variables)
        debug['gradient_norm'] = utils.tf_norm(gradients)
        debug['gradient_global_norm'] = utils.tf_global_norm(debug['gradient_norm'])

        if self.should_clip_gradients:
            gradients, _ = utils.clip_gradients2(gradients, norm=self.clip_norm())
            debug['gradient_clipped_norm'] = utils.tf_norm(gradients)
            debug['clip_norm'] = self.clip_norm.value

        return debug, gradients

    @tf.function
    def update(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


@Network.register(name='A2C-CriticNetwork')
class CriticNetwork(ValueNetwork):

    def train_step(self, batch: dict):
        if isinstance(batch, tuple):
            batch = batch[0]

        debug, grads = self.train_on_batch(batch)
        self.agent.log(average=True, **({f'{self.prefix}_{k}': v for k, v in debug.items()}))

        return grads

    def objective(self, batch, reduction=tf.reduce_sum) -> tuple:
        return super().objective(batch, reduction=reduction)

    @tf.function
    def train_on_batch(self, batch):
        with tf.GradientTape() as tape:
            loss, debug = self.objective(batch)

        gradients = tape.gradient(loss, self.trainable_variables)
        debug['gradient_norm'] = utils.tf_norm(gradients)
        debug['gradient_global_norm'] = utils.tf_global_norm(debug['gradient_norm'])

        if self.should_clip_gradients:
            gradients, _ = utils.clip_gradients2(gradients, norm=self.clip_norm())
            debug['gradient_clipped_norm'] = utils.tf_norm(gradients)
            debug['clip_norm'] = self.clip_norm.value

        return debug, gradients

    @tf.function
    def update(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class ParallelGAEMemory(EpisodicMemory):

    def __init__(self, *args, agent: A2C, **kwargs):
        super().__init__(*args, **kwargs)

        if 'return' in self.data:
            raise ValueError('Key "return" is reserved!')

        if 'advantage' in self.data:
            raise ValueError('Key "advantage" is reserved!')

        self.data['return'] = np.zeros_like(self.data['value'])
        self.data['advantage'] = np.zeros(shape=self.shape + (1,), dtype=np.float32)
        self.agent = agent

    def is_full(self) -> bool:
        if self.full:
            return True

        if self.index * self.shape[0] >= self.size:
            self.full = True

        return self.full

    def _store(self, data, spec, key, value):
        if not isinstance(value, dict):
            array = np.asanyarray(value, dtype=np.float32)
            array = np.reshape(array, newshape=(self.shape[0], spec['shape'][-1]))  # TODO: check spec['shape']

            # indexing: key, env, index (timestep)
            #   - `array` has shape (n_envs, horizon)
            #   - each `v` in `array` is data for the corresponding env
            for env_index, v in enumerate(array):
                data[key][env_index][self.index] = v
        else:
            for k, v in value.items():
                self._store(data=data[key], spec=spec[k], key=k, value=v)

    def end_trajectory(self, last_values) -> dict:
        debug = dict()
        data_reward, data_value = self.data['reward'], self.data['value']
        data_return, data_adv = self.data['return'], self.data['advantage']

        for i in range(self.agent.num_actors):
            v = tf.expand_dims(last_values[i], axis=-1)
            rewards = np.concatenate([data_reward[i][:self.index], v], axis=0)
            values = np.concatenate([data_value[i][:self.index], v], axis=0)

            # compute returns and advantages for i-th environment
            returns = self.compute_returns(rewards)
            adv, advantages = self.compute_advantages(rewards, values)

            # store them
            data_return[i][:self.index] = returns
            data_adv[i][:self.index] = advantages

            # debug
            debug[f'returns_{i}'] = returns
            debug[f'advantages_normalized_{i}'] = advantages
            debug[f'advantages_{i}'] = adv
            debug[f'values_{i}'] = values

        return debug

    def compute_returns(self, rewards):
        returns = utils.rewards_to_go(rewards, discount=self.agent.gamma)
        return returns

    def compute_advantages(self, rewards, values):
        advantages = utils.gae(rewards, values=values, gamma=self.agent.gamma, lambda_=self.agent.lambda_)
        norm_adv = self.agent.adv_normalization_fn(advantages) * self.agent.adv_scale()
        return advantages, norm_adv

    def get_data(self) -> List[dict]:
        """Returns a batch of data for each environment/actor"""
        if self.full:
            index = self.agent.horizon
        else:
            index = self.index

        n_envs = self.agent.num_actors

        def _get(data_list, _k, _v):
            if not isinstance(_v, dict):
                for i in range(n_envs):
                    data_list[i][_k] = _v[i][:index]
            else:
                for data in data_list:
                    data[_k] = dict()

                for k, v in _v.items():
                    _get([data[_k] for data in data_list], k, v)

        batches = [dict() for _ in range(n_envs)]

        for key, value in self.data.items():
            _get(batches, key, value)

        return batches


if __name__ == '__main__':
    a2c = A2C(env='CartPole-v0', horizon=5, use_summary=True, seed=42)
    a2c.learn(250, 200)
