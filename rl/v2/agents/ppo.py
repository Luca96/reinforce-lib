"""Proximal Policy Optimization (PPO)"""

import os
import time
import tensorflow as tf
import numpy as np

from rl import utils
from rl.parameters import DynamicParameter

from rl.v2.agents import Agent, ParallelAgent
from rl.v2.memories import EpisodicMemory, TransitionSpec
from rl.v2.networks import PolicyNetwork, DecomposedValueNetwork
from rl.v2.agents.a2c import ParallelGAEMemory

from typing import Dict, Tuple, Union


# TODO: should memory be reset on episode end?
# TODO: parallel environments
class PPO(Agent):
    def __init__(self, *args, name='ppo-agent', policy_lr: utils.DynamicType = 1e-3, value_lr: utils.DynamicType = 3e-4,
                 optimization_steps=(1, 1), lambda_=0.95, optimizer='adam', load=False, policy: dict = None,
                 clip_ratio: utils.DynamicType = 0.2, clip_norm: Tuple[utils.DynamicType] = (1.0, 1.0), memory_size=8,
                 entropy_strength: utils.DynamicType = 0.01, advantage_scale: utils.DynamicType = 2.0,
                 normalize_advantages: Union[None, str] = 'sign', value: dict = None, **kwargs):
        assert lambda_ > 0.0
        assert memory_size >= 1
        assert max(optimization_steps) <= memory_size

        super().__init__(*args, name=name, **kwargs)

        self.memory_episodes = memory_size
        self.memory_size = 0  # init on learn()

        # Hyper-parameters:
        self.optimization_steps = optimization_steps
        self.lambda_ = tf.constant(lambda_, dtype=tf.float32)
        self.clip_ratio = DynamicParameter.create(value=clip_ratio)
        self.entropy_strength = DynamicParameter.create(value=entropy_strength)
        self.adv_scale = DynamicParameter.create(value=advantage_scale)
        self.adv_normalization_fn = utils.get_normalization_fn(name=normalize_advantages)

        self.weights_path = dict(policy=os.path.join(self.base_path, 'policy'),
                                 value=os.path.join(self.base_path, 'value'))

        self.policy_lr = DynamicParameter.create(value=policy_lr)
        self.value_lr = DynamicParameter.create(value=value_lr)

        # Networks (actor = policy; critic = value)
        self.policy = PolicyNet(agent=self, **(policy or {}))
        self.value = DecomposedValueNetwork(agent=self, **(value or {}))

        self.policy.compile(optimizer, clip_norm=clip_norm[0], learning_rate=self.policy_lr)
        self.value.compile(optimizer, clip_norm=clip_norm[1], learning_rate=self.value_lr)

        if load:
            self.load()

    @property
    def transition_spec(self) -> TransitionSpec:
        return TransitionSpec(state=self.state_spec, action=(self.num_actions,), next_state=False, terminal=False,
                              other=dict(log_prob=(self.num_actions,), value=(2,)))

    @property
    def memory(self) -> 'GAEMemory':
        if self._memory is None:
            self._memory = GAEMemory(self.transition_spec, agent=self, size=self.memory_size)

        return self._memory

    # @tf.function
    def act(self, state) -> Tuple[tf.Tensor, dict, dict]:
        action, log_prob, mean, std = self.policy(state, training=False)
        value = self.value(state, training=False)

        other = dict(log_prob=log_prob, value=value)
        debug = dict(distribution_mean=mean, distribution_std=std)

        return action, other, debug

    def update(self):
        if not self.memory.full_enough(amount=max(self.optimization_steps) * self.batch_size):
            print('Not updated: memory not yet full!')
            return

        t0 = time.time()

        # Prepare data:
        data = self.memory.get_data()

        self.policy.fit(x=data, y=None, batch_size=self.batch_size, steps_per_epoch=self.optimization_steps[0],
                        shuffle=True, verbose=0)
        self.value.fit(x=data, y=None, batch_size=self.batch_size, steps_per_epoch=self.optimization_steps[1],
                       shuffle=True, verbose=0)

        print(f'Update in {round(time.time() - t0, 3)}s')

    def learn(self, episodes: int, timesteps: int, **kwargs):
        # TODO(bug): if `learn()` is called again with different `timesteps` as `GAEMemory` is defined
        t0 = time.time()
        self.memory_size = self.memory_episodes * timesteps

        super().learn(episodes, timesteps, **kwargs)
        print(f'Training took {round(time.time() - t0, 3)}s.')

    def on_termination(self, last_transition, timestep: int, episode: int):
        super().on_termination(last_transition, timestep, episode)

        if last_transition['terminal']:
            value = tf.zeros(shape=(1, 2), dtype=tf.float32)
        else:
            terminal_state = self.preprocess(state=last_transition['next_state'])
            value = self.value(terminal_state, training=False)

        debug = self.memory.end_trajectory(last_value=value)
        self.log(average=True, **debug)

        self.update()

    def load_weights(self):
        self.policy.load_weights(filepath=self.weights_path['policy'], by_name=False)
        self.value.load_weights(filepath=self.weights_path['value'], by_name=False)

    def save_weights(self):
        self.policy.save_weights(filepath=self.weights_path['policy'])
        self.value.save_weights(filepath=self.weights_path['value'])

    def summary(self):
        self.policy.summary()
        self.value.summary()


class PolicyNet(PolicyNetwork):

    @tf.function
    def objective(self, batch, reduction=tf.reduce_mean) -> tuple:
        advantages = batch['advantage']
        old_log_prob = batch['log_prob']

        new_log_prob, entropy = self(batch['state'], actions=batch['action'], training=True)

        # TODO: kl-divergence?

        # Entropy
        entropy = reduction(entropy)
        entropy_penalty = entropy * self.agent.entropy_strength()

        # Probability ratio
        ratio = tf.math.exp(new_log_prob - old_log_prob)
        ratio = reduction(ratio, axis=1)  # per-action mean ratio
        ratio = tf.expand_dims(ratio, axis=-1)

        # Clipped ratio times advantage
        clip_value = self.agent.clip_ratio()
        min_adv = tf.where(advantages > 0.0, x=(1.0 + clip_value) * advantages, y=(1.0 - clip_value) * advantages)

        # Loss
        policy_loss = -reduction(tf.minimum(ratio * advantages, min_adv))
        total_loss = policy_loss - entropy_penalty

        # Debug
        debug = dict(ratio=ratio, log_prob=new_log_prob, old_log_prob=old_log_prob, entropy=entropy,
                     loss=policy_loss, ratio_clip=clip_value, loss_entropy=entropy_penalty, loss_total=total_loss)

        return total_loss, debug


class GAEMemory(EpisodicMemory):

    def __init__(self, *args, agent: PPO, **kwargs):
        super().__init__(*args, **kwargs)

        if 'return' in self.data:
            raise ValueError('Key "return" is reserved!')

        if 'advantage' in self.data:
            raise ValueError('Key "advantage" is reserved!')

        self.data['return'] = np.zeros_like(self.data['value'])
        self.data['advantage'] = np.zeros(shape=(self.size, 1), dtype=np.float32)

        self.last_index = 0
        self.agent = agent

    def end_trajectory(self, last_value: tf.Tensor):
        value = last_value[:, 0] * tf.pow(10.0, last_value[:, 1])
        value = tf.expand_dims(value, axis=-1)

        data_reward = self.data['reward']
        data_value = self.data['value']

        if self.last_index > self.index:
            rewards = tf.concat([data_reward[self.last_index:], data_reward[:self.index], value], axis=0)
            values = tf.concat([data_value[self.last_index:], data_value[:self.index], last_value], axis=0)
        else:
            rewards = tf.concat([data_reward[self.last_index:self.index], value], axis=0)
            values = tf.concat([data_value[self.last_index:self.index], last_value], axis=0)

        # value = base * 10^exponent
        v_base, v_exp = values[:, 0], values[:, 1]
        values = v_base * tf.pow(10.0, v_exp)
        values = tf.expand_dims(values, axis=-1)

        # compute returns and advantages for current episode
        returns = self.compute_returns(rewards)
        adv, advantages = self.compute_advantages(rewards, values)

        # store them
        if self.last_index > self.index:
            k = self.size - self.last_index

            self.data['return'][self.last_index:] = returns[:k]
            self.data['return'][:self.index] = returns[k:]

            self.data['advantage'][self.last_index:] = advantages[:k]
            self.data['advantage'][:self.index] = advantages[k:]
        else:
            self.data['return'][self.last_index:self.index] = returns
            self.data['advantage'][self.last_index:self.index] = advantages

        # update index
        self.last_index = self.index

        # debug
        return dict(returns=returns[:, 0] * tf.pow(10.0, returns[:, 1]), advantages_normalized=advantages,
                    advantages=adv, values_base=v_base, values=values, returns_base=returns[:, 0],
                    returns_exp=returns[:, 1], values_exp=v_exp)

    def compute_returns(self, rewards: tf.Tensor):
        returns = utils.rewards_to_go(rewards, discount=self.agent.gamma)
        returns = utils.to_float(returns)

        returns = tf.map_fn(fn=utils.decompose_number, elems=returns, dtype=(tf.float32, tf.float32))
        returns = tf.concat([returns[0], tf.reshape(returns[1], shape=returns[0].shape)], axis=-1)
        return returns

    def compute_advantages(self, rewards: tf.Tensor, values: tf.Tensor):
        advantages = utils.gae(rewards, values=values, gamma=self.agent.gamma, lambda_=self.agent.lambda_)
        norm_adv = self.agent.adv_normalization_fn(advantages) * self.agent.adv_scale()
        return advantages, norm_adv


# TODO: jointly optimize `policy` and `value` networks, such that \pi does not have outdated values estimates
# TODO: share features among the two networks??
# TODO(bug): probable memory leak!
class PPO2(ParallelAgent):
    def __init__(self, env, horizon: int, batch_size: int, optimization_epochs=(1, 1), gamma=0.99, load=False,
                 policy_lr: utils.DynamicType = 1e-3, value_lr: utils.DynamicType = 3e-4, optimizer='adam',
                 lambda_=0.95, num_actors=16, name='ppo2-agent', clip_ratio: utils.DynamicType = 0.2,
                 policy: dict = None, value: dict = None, entropy: utils.DynamicType = 0.01, clip_norm=(1.0, 1.0),
                 advantage_scale: utils.DynamicType = 2.0, normalize_advantages: Union[None, str] = 'sign', **kwargs):
        assert horizon >= 1

        super().__init__(env, num_actors=num_actors, batch_size=batch_size, gamma=gamma, name=name, **kwargs)

        # Hyper-parameters:
        self.horizon = int(horizon)
        self.opt_epochs = self._init_optimization_epochs(optimization_epochs)
        self.lambda_ = tf.constant(lambda_, dtype=tf.float32)
        self.clip_ratio = DynamicParameter.create(value=clip_ratio)
        self.entropy_strength = DynamicParameter.create(value=entropy)
        self.adv_scale = DynamicParameter.create(value=advantage_scale)
        self.adv_normalization_fn = utils.get_normalization_fn(name=normalize_advantages)

        self.policy_lr = DynamicParameter.create(value=policy_lr)
        self.value_lr = DynamicParameter.create(value=value_lr)

        # Networks
        self.weights_path = dict(policy=os.path.join(self.base_path, 'policy'),
                                 value=os.path.join(self.base_path, 'value'))

        self.policy = PolicyNet(agent=self, **(policy or {}))
        self.value = DecomposedValueNetwork(agent=self, **(value or {}))

        self.policy.compile(optimizer, clip_norm=clip_norm[0], learning_rate=self.policy_lr)
        self.value.compile(optimizer, clip_norm=clip_norm[1], learning_rate=self.value_lr)

        if load:
            self.load()

    def _init_optimization_epochs(self, opt_epochs: Union[tuple, utils.DynamicType]) -> dict:
        if isinstance(opt_epochs, int) or isinstance(opt_epochs, float):
            return dict(policy=DynamicParameter.create(value=int(opt_epochs)),
                        value=DynamicParameter.create(value=int(opt_epochs)))

        if isinstance(opt_epochs, tuple):
            assert 1 <= len(opt_epochs) <= 2

            if len(opt_epochs) == 1:
                opt_epochs = opt_epochs[0]
                return dict(policy=DynamicParameter.create(value=int(opt_epochs)),
                            value=DynamicParameter.create(value=int(opt_epochs)))
            else:
                return dict(policy=DynamicParameter.create(value=int(opt_epochs[0])),
                            value=DynamicParameter.create(value=int(opt_epochs[1])))
        else:
            # default value
            return self._init_optimization_epochs(opt_epochs=(1, 1))

    @property
    def transition_spec(self) -> TransitionSpec:
        state_spec = {k: (self.horizon,) + shape for k, shape in self.state_spec.items()}

        return TransitionSpec(state=state_spec, action=(self.horizon, self.num_actions), next_state=False,
                              terminal=False, reward=(self.horizon, 1),
                              other=dict(log_prob=(self.horizon, self.num_actions), value=(self.horizon, 2)))

    @property
    def memory(self) -> 'GAEMemory2':
        if self._memory is None:
            self._memory = GAEMemory2(self.transition_spec, agent=self, size=self.num_actors)

        return self._memory

    def act(self, states) -> Tuple[tf.Tensor, dict, dict]:
        actions, log_prob, mean, std = self.policy(states, training=False)
        values = self.value(states, training=False)

        other = dict(log_prob=log_prob, value=values)
        debug = dict()

        if self.distribution_type != 'categorical':
            for i, (mu, sigma) in enumerate(zip(mean, std)):
                debug[f'distribution_mean_{i}'] = mu
                debug[f'distribution_std_{i}'] = sigma

        return actions, other, debug

    def update(self):
        if not self.memory.full_enough(amount=self.batch_size):
            print('Not updated: memory not enough full!')
            return

        t0 = time.time()
        data = self.memory.get_data()

        self.policy.fit(x=data, y=None, batch_size=self.batch_size, epochs=self.opt_epochs['policy'](),
                        shuffle=True, verbose=0)
        self.value.fit(x=data, y=None, batch_size=self.batch_size, epochs=self.opt_epochs['value'](),
                       shuffle=True, verbose=0)

        print(f'Update in {round(time.time() - t0, 3)}s')

    def on_transition(self, transition: Dict[str, list], timestep: int, episode: int):
        super().on_transition(transition, timestep, episode)

        if any(transition['terminal']) or (timestep % self.horizon == 0) or (timestep == self.max_timesteps):
            terminal_states = self.preprocess(transition['next_state'])

            values = self.value(terminal_states, training=False)
            values = values * utils.to_float(tf.logical_not(transition['terminal']))

            debug = self.memory.end_trajectory(last_values=values)
            self.log(average=True, **debug)

            self.update()
            self.memory.clear()

    def load_weights(self):
        self.policy.load_weights(filepath=self.weights_path['policy'], by_name=False)
        self.value.load_weights(filepath=self.weights_path['value'], by_name=False)

    def save_weights(self):
        self.policy.save_weights(filepath=self.weights_path['policy'])
        self.value.save_weights(filepath=self.weights_path['value'])

    def summary(self):
        self.policy.summary()
        self.value.summary()


class GAEMemory2(ParallelGAEMemory):

    def full_enough(self, amount: int) -> bool:
        return self.full or self.index * self.agent.num_actors >= amount

    def end_trajectory(self, last_values: tf.Tensor):
        last_v = last_values[:, 0] * tf.pow(10.0, last_values[:, 1])

        debug = dict()
        data_reward, data_value = self.data['reward'], self.data['value']
        data_return, data_adv = self.data['return'], self.data['advantage']

        for i in range(self.agent.num_actors):
            last_value = tf.expand_dims(last_values[i], axis=0)

            rewards = np.concatenate([data_reward[i][:self.index], tf.reshape(last_v[i], shape=(1, 1))], axis=0)
            values = np.concatenate([data_value[i][:self.index], last_value], axis=0)

            # value = base * 10^exponent
            v_base, v_exp = values[:, 0], values[:, 1]
            values = v_base * tf.pow(10.0, v_exp)
            values = tf.expand_dims(values, axis=-1)

            # compute returns and advantages for current episode
            returns = self.compute_returns(rewards)
            adv, advantages = self.compute_advantages(rewards, values)

            # store them
            data_return[i][:self.index] = returns
            data_adv[i][:self.index] = advantages

            # debug
            debug.update({f'returns_{i}': returns[:, 0] * tf.pow(10.0, returns[:, 1]),
                          f'returns_base_{i}': returns[:, 0], f'returns_exp_{i}': returns[:, 1],
                          f'advantages_normalized_{i}': advantages, f'advantages_{i}': adv,
                          f'values_{i}': values, f'values_base_{i}': v_base, f'values_exp_{i}': v_exp})
        return debug

    def compute_returns(self, rewards):
        returns = utils.rewards_to_go(rewards, discount=self.agent.gamma)
        returns = utils.to_float(returns)

        returns = tf.map_fn(fn=utils.decompose_number, elems=returns, dtype=(tf.float32, tf.float32))
        returns = tf.concat([returns[0], tf.reshape(returns[1], shape=returns[0].shape)], axis=-1)
        return returns

    def get_data(self) -> dict:
        if self.full:
            index = self.size
        else:
            index = self.index

        def get(data_, key_, val):
            if not isinstance(val, dict):
                v = np.stack([val[i][:index] for i in range(self.agent.num_actors)])
                v = np.reshape(v, newshape=(v.shape[0] * v.shape[1],) + v.shape[2:])  # concat over actors
                data_[key_] = v
            else:
                data_[key_] = dict()

                for k, v in val.items():
                    get(data_[key_], k, v)

        data = dict()

        for key, value in self.data.items():
            get(data, key, value)

        return data


def lunar_lander():
    policy = dict(activation=tf.nn.swish, num_layers=4, units=64)
    value = dict(activation=tf.nn.tanh, num_layers=4, units=64, exponent_scale=4.0)

    a = PPO(env='LunarLanderContinuous-v2', name='ppo-lunar', batch_size=64, gamma=1.0, lambda_=1.0, memory_size=8,
            entropy_strength=0.001,
            optimization_steps=(4, 4), policy_lr=3e-4, policy=policy, value=value, seed=42)
    a.summary()
    a.learn(1000, 200, evaluation=dict(freq=50, episodes=10))


def cartpole():
    policy = dict(activation=tf.nn.swish)
    value = dict(activation=tf.nn.swish, exponent_scale=3.0)

    a = PPO(env='CartPole-v0', batch_size=32, gamma=1.0, lambda_=1.0, memory_size=4, entropy_strength=0.0,
            optimization_steps=(2, 1), policy_lr=3e-4, policy=policy, value=value, seed=42)
    a.learn(200, 200)


def cartpole2():
    policy = dict(activation=tf.nn.swish, units=32, bias_initializer='glorot_uniform',
                  kernel_initializer='glorot_normal')
    value = dict(activation=tf.nn.tanh, exponent_scale=3.0, bias_initializer='glorot_uniform',
                 kernel_initializer='glorot_normal')

    a = PPO2(env='CartPole-v0', batch_size=32, horizon=200, num_actors=16//2, lambda_=0.95, entropy=0.0,
             optimization_epochs=(10, 1), policy_lr=3e-4, policy=policy, value=value, seed=42, use_summary=True)
    a.learn(500, 200)


if __name__ == '__main__':
    # cartpole()
    cartpole2()
    # lunar_lander()
