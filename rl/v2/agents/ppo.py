"""Proximal Policy Optimization (PPO)"""

import os
import gym
import time
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import *

from rl import utils
from rl.parameters import DynamicParameter

from rl.v2.agents import Agent
from rl.v2.memories import EpisodicMemory, TransitionSpec
from rl.v2.networks import Network, backbones

from typing import Dict, Tuple, Union


# TODO: parallel environments
# TODO: check log_prob shape with discrete and continuous actions
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

        self._init_action_space()
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
        self.policy = PolicyNetwork(agent=self, **(policy or {}))
        self.value = ValueNetwork(agent=self, **(value or {}))

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

    def act(self, state) -> Tuple[tf.Tensor, dict, dict]:
        action, log_prob, mean, std = self.policy(state, training=False)
        value = self.value(state, training=False)

        other = dict(log_prob=log_prob, value=value)
        debug = dict(distribution_mean=mean, distribution_std=std)

        return action, other, debug

    def _init_action_space(self):
        action_space = self.env.action_space

        if isinstance(action_space, gym.spaces.Box):
            self.num_actions = action_space.shape[0]

            # continuous:
            if action_space.is_bounded():
                self.distribution_type = 'beta'

                self.action_low = tf.constant(action_space.low, dtype=tf.float32)
                self.action_high = tf.constant(action_space.high, dtype=tf.float32)
                self.action_range = tf.constant(action_space.high - action_space.low, dtype=tf.float32)

                self.convert_action = lambda a: tf.squeeze(a * self.action_range + self.action_low).numpy()
            else:
                self.distribution_type = 'gaussian'
                self.convert_action = lambda a: tf.squeeze(a).numpy()
        else:
            # discrete:
            assert isinstance(action_space, gym.spaces.Discrete)
            self.distribution_type = 'categorical'

            self.num_actions = 1
            self.num_classes = action_space.n
            self.convert_action = lambda a: tf.cast(tf.squeeze(a), dtype=tf.int32).numpy()

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
        # TODO(bug): if `learn()` is called again with different `timesteps`
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


class PolicyNetwork(Network):

    def __init__(self, agent: PPO, eps=utils.EPSILON, log_prefix='policy', **kwargs):
        self._base_model_initialized = True  # weird hack

        self.dist_args = kwargs.pop('distribution', {})
        self.distribution = agent.distribution_type
        self.eps = eps

        super().__init__(agent, target=False, log_prefix=log_prefix, **kwargs)

    @tf.function
    def call(self, inputs, actions=None, training=False, **kwargs):
        policy: tfp.distributions.Distribution = super().call(inputs, training=training, **kwargs)

        new_actions = self._round_actions_if_necessary(actions=policy)
        log_prob = policy.log_prob(new_actions)

        if self.distribution != 'categorical':
            mean = policy.mean()
            std = policy.stddev()
        else:
            mean = std = 0.0

        if tf.is_tensor(actions):
            actions = self._round_actions_if_necessary(actions)

            # compute `log_prob` on given `actions`
            return policy.log_prob(actions), policy.entropy()
        else:
            return new_actions, log_prob, mean, std

    @tf.function
    def _round_actions_if_necessary(self, actions):
        if self.distribution == 'beta':
            # round samples (actions) before computing density:
            # https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Beta
            return tf.clip_by_value(actions, self.eps, 1.0 - self.eps)

        return actions

    def structure(self, inputs: Dict[str, Input], name='PolicyNetwork', **kwargs) -> tuple:
        inputs = inputs['state']
        x = backbones.dense(layer=inputs, **kwargs)

        output = self.output_layer(x)
        return inputs, output, name

    def output_layer(self, layer: Layer) -> Layer:
        return self.get_distribution_layer(layer, **self.dist_args)

    @tf.function
    def objective(self, batch) -> tuple:
        advantages = batch['advantage']
        old_log_prob = batch['log_prob']

        new_log_prob, entropy = self(batch['state'], actions=batch['action'], training=True)

        # TODO: kl-divergence?

        # Entropy
        entropy = tf.reduce_mean(entropy)
        entropy_penalty = entropy * self.agent.entropy_strength()

        # Probability ratio
        ratio = tf.math.exp(new_log_prob - old_log_prob)
        ratio = tf.reduce_mean(ratio, axis=1)  # per-action mean ratio
        ratio = tf.expand_dims(ratio, axis=-1)

        # Clipped ratio times advantage
        clip_value = self.agent.clip_ratio()
        min_adv = tf.where(advantages > 0.0, x=(1.0 + clip_value) * advantages, y=(1.0 - clip_value) * advantages)

        # Loss
        policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, min_adv))
        total_loss = policy_loss - entropy_penalty

        # Debug
        debug = dict(ratio=ratio, log_prob=new_log_prob, old_log_prob=old_log_prob, entropy=entropy,
                     loss=policy_loss, ratio_clip=clip_value, loss_entropy=entropy_penalty, loss_total=total_loss)

        return total_loss, debug

    def get_distribution_layer(self, layer: Layer, min_std=0.02, unimodal=False,
                               **kwargs) -> tfp.layers.DistributionLambda:
        """
        A probability distribution layer, used for sampling actions, computing the `log_prob`, `entropy` ecc.

        :param layer: last layer of the network (e.g. actor, critic, policy networks ecc)
        :param min_std: minimum variance, useful for 'beta' and especially 'gaussian' to prevent NaNs.
        :param unimodal: only used in 'beta' to make it concave and unimodal.
        :param kwargs: additional argument given to tf.keras.layers.Dense.
        :return: tfp.layers.DistributionLambda instance.
        """
        assert min_std >= 0.0

        min_std = tf.constant(min_std, dtype=tf.float32)

        # Discrete actions:
        if self.distribution == 'categorical':
            num_actions = self.agent.num_actions
            num_classes = self.agent.num_classes

            logits = Dense(units=num_actions * num_classes, activation='linear', **kwargs)(layer)
            logits = Reshape((num_actions, num_classes), name='logits')(logits)

            return tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Categorical(logits=t))(logits)

        # Bounded continuous 1-dimensional actions:
        # for activations choice refer to chapter 4 of http://proceedings.mlr.press/v70/chou17a/chou17a.pdf
        if self.distribution == 'beta':
            num_actions = self.agent.num_actions

            if unimodal:
                min_std += 1.0

            alpha = Dense(units=num_actions, activation=utils.softplus(min_std), name='alpha', **kwargs)(layer)
            beta = Dense(units=num_actions, activation=utils.softplus(min_std), name='beta', **kwargs)(layer)

            return tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Beta(t[0], t[1]))([alpha, beta])

        # Unbounded continuous actions)
        # for activations choice see chapter 4 of http://proceedings.mlr.press/v70/chou17a/chou17a.pdf
        if self.distribution == 'gaussian':
            num_actions = self.agent.num_actions

            mu = Dense(units=num_actions, activation='linear', name='mu', **kwargs)(layer)
            sigma = Dense(units=num_actions, activation=utils.softplus(min_std), name='sigma', **kwargs)(layer)

            return tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Normal(loc=t[0], scale=t[1]))([mu, sigma])


class ValueNetwork(Network):

    def __init__(self, agent: PPO, exponent_scale=6.0, target=False, log_prefix='value', normalize_loss=True,
                 **kwargs):
        self._base_model_initialized = True  # weird hack
        self.exp_scale = tf.constant(exponent_scale, dtype=tf.float32)
        self.normalize_loss = bool(normalize_loss)

        super().__init__(agent, target=target, log_prefix=log_prefix, **kwargs)

    @tf.function
    def call(self, *args, **kwargs):
        return super().call(*args, **kwargs)

    def structure(self, inputs: Dict[str, Input], name='ValueNetwork', **kwargs) -> tuple:
        inputs = inputs['state']
        x = backbones.dense(layer=inputs, **kwargs)

        output = self.output_layer(x)
        return inputs, output, name

    def output_layer(self, layer: Layer) -> Layer:
        base = Dense(units=1, activation=tf.nn.tanh, name='v-base')(layer)
        exp = Dense(units=1, activation=lambda x: self.exp_scale * tf.nn.sigmoid(x), name='v-exp')(layer)

        return concatenate([base, exp], axis=1)

    @tf.function
    def objective(self, batch) -> tuple:
        states, returns = batch['state'], batch['return']
        values = self(states, training=True)

        base_loss = 0.5 * tf.reduce_mean(tf.square(returns[:, 0] - values[:, 0]))
        exp_loss = 0.5 * tf.reduce_mean(tf.square(returns[:, 1] - values[:, 1]))

        if self.normalize_loss:
            loss = 0.25 * base_loss + exp_loss / (self.exp_scale ** 2)
        else:
            loss = base_loss + exp_loss

        return loss, dict(loss_base=base_loss, loss_exp=exp_loss, loss=loss)


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
            optimization_steps=(4 // 2, 4 // 2), policy_lr=3e-4, policy=policy, value=value, seed=42)
    a.learn(1000 // 50, 200)


if __name__ == '__main__':
    cartpole()
    # lunar_lander()
