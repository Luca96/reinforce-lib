"""Proximal Policy Optimization (PPO)"""

import os
import tensorflow as tf
import numpy as np

from rl import utils
from rl.parameters import DynamicParameter

from rl.v2.agents import ParallelAgent
from rl.v2.memories import TransitionSpec
from rl.v2.networks import Network, PolicyNetwork, ValueNetwork, DecomposedValueNetwork
from rl.v2.agents.a2c import ParallelGAEMemory

from typing import Dict, Tuple, Union, Callable


# TODO: some performance issue (slow and variable inference speed => try @tf.function on `act`)
# TODO: share features among the two networks??
# TODO(bug): probable memory leak!
class PPO1(ParallelAgent):
    """Vanilla PPO agent"""

    def __init__(self, env, horizon: int, batch_size: int, optimization_epochs=10, gamma=0.99, load=False,
                 policy_lr: utils.DynamicType = 1e-3, value_lr: utils.DynamicType = 3e-4, optimizer='adam',
                 lambda_=0.95, num_actors=16, name='ppo1-agent', clip_ratio: utils.DynamicType = 0.2,
                 policy: dict = None, value: dict = None, entropy: utils.DynamicType = 0.01, clip_norm=(1.0, 1.0),
                 advantage_scale: utils.DynamicType = 1.0, normalize_advantages: Union[None, str, Callable] = None,
                 normalize_returns: Union[None, str, Callable] = None, **kwargs):
        assert horizon >= 1
        assert optimization_epochs >= 1

        super().__init__(env, num_actors=num_actors, batch_size=batch_size, gamma=gamma, name=name, **kwargs)

        # Hyper-parameters:
        self.horizon = int(horizon)
        self.opt_epochs = int(optimization_epochs)
        self.lambda_ = tf.constant(lambda_, dtype=tf.float32)
        self.clip_ratio = DynamicParameter.create(value=clip_ratio)
        self.entropy_strength = DynamicParameter.create(value=entropy)
        self.adv_scale = DynamicParameter.create(value=advantage_scale)

        self.adv_normalization_fn = utils.get_normalization_fn(arg=normalize_advantages)
        self.returns_norm_fn = utils.get_normalization_fn(arg=normalize_returns)

        self.policy_lr = DynamicParameter.create(value=policy_lr)
        self.value_lr = DynamicParameter.create(value=value_lr)

        # Networks
        self.weights_path = dict(policy=os.path.join(self.base_path, 'policy'),
                                 value=os.path.join(self.base_path, 'value'))

        self.policy = Network.create(agent=self, **(policy or {}), base_class='PPO-PolicyNetwork')
        self.value = Network.create(agent=self, **(value or {}), base_class=ValueNetwork)

        self.policy.compile(optimizer, clip_norm=clip_norm[0], learning_rate=self.policy_lr)
        self.value.compile(optimizer, clip_norm=clip_norm[1], learning_rate=self.value_lr)

        if load:
            self.load()

    @property
    def transition_spec(self) -> TransitionSpec:
        return TransitionSpec(state=self.state_spec, action=(self.num_actions,), next_state=False, terminal=False,
                              reward=(1,), other=dict(log_prob=(self.num_actions,), value=(1,)))

    @property
    def memory(self) -> 'GAEMemory1':
        if self._memory is None:
            self._memory = GAEMemory1(self.transition_spec, agent=self, shape=(self.num_actors, self.horizon))

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

    # TODO: implement early stop based on kl-divergence?
    def update(self):
        if not self.memory.full_enough(amount=self.batch_size):
            return self.memory.update_warning(self.batch_size)

        with utils.Timed('Update'):
            batches = self.memory.to_batches(repeat=self.opt_epochs, **self.data_args)

            for batch in batches:
                self.policy.train_step(batch)
                self.value.train_step(batch)

            self.memory.clear()

    def on_transition(self, transition: Dict[str, list], timestep: int, episode: int):
        super().on_transition(transition, timestep, episode)

        if any(transition['terminal']) or (timestep % self.horizon == 0) or (timestep == self.max_timesteps) \
                or self.memory.is_full():
            terminal_states = self.preprocess(transition['next_state'])

            values = self.value(terminal_states, training=False)
            values = values * utils.to_float(tf.logical_not(transition['terminal']))

            debug = self.memory.end_trajectory(last_values=values)
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


@Network.register(name='PPO-PolicyNetwork')
class PolicyNet(PolicyNetwork):

    @tf.function
    def objective(self, batch, reduction=tf.reduce_mean) -> tuple:
        advantages = batch['advantage']
        old_log_prob = batch['log_prob']

        new_log_prob, entropy = self(batch['state'], actions=batch['action'], training=True)

        # TODO: kl-divergence?
        kld = utils.kl_divergence(new_log_prob, old_log_prob)

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
        debug = dict(ratio=ratio, log_prob=new_log_prob, old_log_prob=old_log_prob, entropy=entropy, kl_divergence=kld,
                     loss=policy_loss, ratio_clip=clip_value, loss_entropy=entropy_penalty, loss_total=total_loss)

        return total_loss, debug


class GAEMemory1(ParallelGAEMemory):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start = 0

    def full_enough(self, amount: int) -> bool:
        return self.full or self.index * self.shape[0] >= amount

    def end_trajectory(self, last_values) -> dict:
        debug = dict()
        data_reward, data_value = self.data['reward'], self.data['value']
        data_return, data_adv = self.data['return'], self.data['advantage']

        for i in range(self.agent.num_actors):
            v = tf.expand_dims(last_values[i], axis=-1)
            rewards = np.concatenate([data_reward[i][self.start:self.index], v], axis=0)
            values = np.concatenate([data_value[i][self.start:self.index], v], axis=0)

            # compute returns and advantages for i-th environment
            returns, returns_norm = self.compute_returns(rewards)
            adv, advantages = self.compute_advantages(rewards, values)

            # store them
            data_return[i][self.start:self.index] = returns_norm
            data_adv[i][self.start:self.index] = advantages

            # debug
            debug[f'returns_{i}'] = returns
            debug[f'returns_norm_{i}'] = returns_norm
            debug[f'advantages_normalized_{i}'] = advantages
            debug[f'advantages_{i}'] = adv
            debug[f'values_{i}'] = values

        self.start = self.index
        return debug

    def clear(self):
        super().clear()
        self.start = 0

    def compute_returns(self, rewards):
        returns = utils.rewards_to_go(rewards, discount=self.agent.gamma)
        return returns, self.agent.returns_norm_fn(returns)

    def get_data(self) -> dict:
        if self.full:
            index = self.shape[1]  # shape[1] = agent.horizon
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


class PPO2(PPO1):
    def __init__(self, *args, name='ppo2-agent', normalize_advantages: Union[None, str] = 'sign', **kwargs):
        value = kwargs.pop('value', {})

        if 'class' not in value:
            value['class'] = DecomposedValueNetwork

        super().__init__(*args, name=name, normalize_advantages=normalize_advantages, value=value, **kwargs)
        assert isinstance(self.value, DecomposedValueNetwork)

    @property
    def transition_spec(self) -> TransitionSpec:
        return TransitionSpec(state=self.state_spec, action=(self.num_actions,), next_state=False,
                              terminal=False, reward=(1,),
                              other=dict(log_prob=(self.num_actions,), value=(2,)))

    @property
    def memory(self) -> 'GAEMemory2':
        if self._memory is None:
            self._memory = GAEMemory2(self.transition_spec, agent=self, shape=(self.num_actors, self.horizon))

        return self._memory


class GAEMemory2(GAEMemory1):

    def end_trajectory(self, last_values: tf.Tensor):
        last_v = last_values[:, 0] * tf.pow(10.0, last_values[:, 1])

        debug = dict()
        data_reward, data_value = self.data['reward'], self.data['value']
        data_return, data_adv = self.data['return'], self.data['advantage']

        for i in range(self.agent.num_actors):
            last_value = tf.expand_dims(last_values[i], axis=0)

            rewards = np.concatenate([data_reward[i][self.start:self.index], tf.reshape(last_v[i], shape=(1, 1))], axis=0)
            values = np.concatenate([data_value[i][self.start:self.index], last_value], axis=0)

            # value = base * 10^exponent
            v_base, v_exp = values[:, 0], values[:, 1]
            values = v_base * tf.pow(10.0, v_exp)
            values = tf.expand_dims(values, axis=-1)

            # compute returns and advantages for current episode
            returns = self.compute_returns(rewards)
            adv, advantages = self.compute_advantages(rewards, values)

            # store them
            data_return[i][self.start:self.index] = returns
            data_adv[i][self.start:self.index] = advantages

            # debug
            debug.update({f'returns_{i}': returns[:, 0] * tf.pow(10.0, returns[:, 1]),
                          f'returns_base_{i}': returns[:, 0], f'returns_exp_{i}': returns[:, 1],
                          f'advantages_normalized_{i}': advantages, f'advantages_{i}': adv,
                          f'values_{i}': values, f'values_base_{i}': v_base, f'values_exp_{i}': v_exp})

        self.start = self.index
        return debug

    def compute_returns(self, rewards):
        returns = utils.rewards_to_go(rewards, discount=self.agent.gamma)
        returns = utils.to_float(returns)

        returns = tf.map_fn(fn=utils.decompose_number, elems=returns, dtype=(tf.float32, tf.float32))
        returns = tf.concat([returns[0], tf.reshape(returns[1], shape=returns[0].shape)], axis=-1)
        return returns


def cartpole1():
    policy = dict(activation=tf.nn.swish, units=32, bias_initializer='glorot_uniform',
                  kernel_initializer='glorot_normal')
    value = dict(activation=tf.nn.tanh, bias_initializer='glorot_uniform', kernel_initializer='glorot_normal')

    args1 = dict(batch_size=32, horizon=200, num_actors=1, lambda_=0.95, entropy=0.0,
                 advantage_scale=2.0, optimization_epochs=10, policy_lr=3e-4)

    args2 = args1.copy()
    args2.update(advantage_scale=1.0, normalize_advantages='magnitude')

    args3: dict = args2.copy()
    args3.update(normalize_advantages=None, normalize_returns='magnitude')

    a = PPO1(env='CartPole-v0', name='ppo1-cart', **args3, policy=policy, value=value, seed=42, use_summary=True)
    a.learn(500, 200)


def lunar1():
    policy = dict(activation=tf.nn.swish, num_layers=4, units=64)
    value = dict(activation=tf.nn.tanh, num_layers=4, units=64, exponent_scale=4.0)

    a = PPO1(env='LunarLanderContinuous-v2', batch_size=32, horizon=200, num_actors=4, lambda_=0.95,
             entropy=0.001, name='ppo1-lunar',
             optimization_epochs=10, policy_lr=3e-4, policy=policy, value=value, seed=42, use_summary=True)
    a.learn(500, 200, render=25)


def cartpole2():
    policy = dict(activation=tf.nn.swish, units=32, bias_initializer='glorot_uniform',
                  kernel_initializer='glorot_normal')
    value = dict(activation=tf.nn.tanh, exponent_scale=3.0, bias_initializer='glorot_uniform',
                 kernel_initializer='glorot_normal')

    a = PPO2(env='CartPole-v0', batch_size=32, horizon=200, num_actors=1, lambda_=0.95, entropy=0.0,
             name='ppo2-cart',
             optimization_epochs=10, policy_lr=3e-4, policy=policy, value=value, seed=42, use_summary=True)
    a.learn(500, 200)


def lunar2():
    policy = dict(activation=tf.nn.swish, num_layers=4, units=64)
    value = dict(activation=tf.nn.tanh, num_layers=4, units=64, exponent_scale=4.0)

    a = PPO2(env='LunarLanderContinuous-v2', batch_size=32, horizon=200, num_actors=4, lambda_=0.95,
             entropy=0.001, name='ppo2-lunar',
             optimization_epochs=10, policy_lr=3e-4, policy=policy, value=value, seed=42, use_summary=True)
    a.learn(500, 200, render=25)


if __name__ == '__main__':
    cartpole1()
    # lunar_lander1()

    # cartpole2()
    # lunar2()
