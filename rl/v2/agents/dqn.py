"""Deep Q-Learning (DQN) with Experience Replay"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import gym
import numpy as np
import tensorflow as tf

from typing import Union, List, Dict, Tuple

from rl import utils
from rl.parameters import DynamicParameter

from rl.v2.agents import Agent
from rl.v2.memories import TransitionSpec, ReplayMemory, NStepMemory, PrioritizedMemory
from rl.v2.networks.q import Network, QNetwork, DoubleQNetwork


class DQN(Agent):
    # TODO: non-terminating summary issue
    def __init__(self, *args, name='dqn-cart_v1', lr: utils.DynamicType = 3e-4, optimizer: Union[dict, str] = 'adam',
                 policy='e-greedy', epsilon: utils.DynamicType = 0.05, clip_norm: utils.DynamicType = None,
                 update_target_network: Union[bool, int] = False, polyak: utils.DynamicType = 0.995, double=True,
                 network: dict = None, dueling=True, horizon=1, prioritized=False, alpha: utils.DynamicType = 0.6,
                 beta: utils.DynamicType = 0.1, memory_size=1024, **kwargs):
        assert horizon >= 1
        assert policy.lower() in ['boltzmann', 'boltzmann2', 'softmax', 'e-greedy', 'greedy']
        super().__init__(*args, name=name, **kwargs)

        self.memory_size = int(memory_size)
        self.policy_fn = self._init_policy(policy=policy.lower())
        self.epsilon = DynamicParameter.create(value=epsilon)  # or `temperature` for "softmax" and "boltzmann"
        self.polyak = DynamicParameter.create(value=polyak)
        self.prioritized = bool(prioritized)
        self.horizon = int(horizon)
        self.rng = utils.get_random_generator(seed=self.seed)

        # PER memory params:
        if self.prioritized:
            self.alpha = DynamicParameter.create(value=alpha)
            self.beta = DynamicParameter.create(value=beta)

        if not update_target_network and self.polyak.value == 1.0:
            self.should_update_target = False
        else:
            self.should_update_target = True
            self.update_target_freq = int(update_target_network)
            self._update_target_freq = self.update_target_freq  # copy

        self.lr = DynamicParameter.create(value=lr)

        self.weights_path = dict(dqn=os.path.join(self.base_path, 'dqn'))

        if double:
            self.dqn = Network.create(agent=self, dueling=dueling, prioritized=self.prioritized, **(network or {}),
                                      base_class=DoubleQNetwork)
        else:
            self.dqn = Network.create(agent=self, dueling=dueling, prioritized=self.prioritized, **(network or {}),
                                      base_class=QNetwork)

        self.dqn.compile(optimizer, clip_norm=clip_norm, clip=self.clip_grads, learning_rate=self.lr)

    @property
    def transition_spec(self) -> TransitionSpec:
        return TransitionSpec(state=self.state_spec, next_state=True, action=(self.num_actions,))

    def define_memory(self) -> Union[ReplayMemory, PrioritizedMemory]:
        if self.prioritized:
            return PrioritizedMemory(self.transition_spec, shape=self.memory_size, gamma=self.gamma,
                                     alpha=self.alpha, beta=self.beta, seed=self.seed)

        return NStepMemory(self.transition_spec, shape=self.memory_size, gamma=self.gamma, seed=self.seed)

    # TODO: add `MultiDiscrete/Binary` action-space support
    def _init_action_space(self):
        assert isinstance(self.env.action_space, gym.spaces.Discrete)

        self.num_actions = 1
        self.num_classes = self.env.action_space.n

        self.convert_action = lambda a: tf.cast(tf.squeeze(a), dtype=tf.int32).numpy()

    def _init_policy(self, policy: str):
        if policy == 'boltzmann':
            return self._boltzmann_policy

        elif policy == 'softmax':
            return self._softmax_policy

        elif policy == 'e-greedy':
            return self._epsilon_greedy_policy

        return self._deterministic_policy

    def act(self, state, deterministic=False, **kwargs) -> Tuple[tf.Tensor, dict, dict]:
        if deterministic:
            return self._deterministic_policy(state, **kwargs)

        return self.policy_fn(state, **kwargs)

    @tf.function
    def _boltzmann_policy(self, state, **kwargs):
        """Boltzmann policy with tricks for numerical stability"""
        q_values = self.dqn.q_values(state, **kwargs)
        q_max = tf.reduce_max(q_values)

        if q_max >= 0.0:
            if q_max <= utils.TF_EPS:
                logits = tf.exp(q_values / (q_max + 1.0))
            else:
                logits = tf.exp(q_values)
        else:
            # negative
            if q_max >= -1:
                logits = tf.exp(q_values - (q_max + 1.0))
            else:
                logits = tf.exp(q_values)

        action = tf.random.categorical(logits=logits, num_samples=1, seed=self.seed)
        return action, {}, dict(exp_q_values=logits)
    
    # TODO: test vs `boltzmann`
    @tf.function
    def _boltzmann2_policy(self, state, **kwargs):
        q_values = self.dqn.q_values(state, **kwargs)
        temperature = self.epsilon()

        # normalize `q_values` for numerical stability
        logits = q_values / (temperature + utils.TF_EPS)
        logits = logits - tf.reduce_max(logits)

        action = tf.random.categorical(logits=tf.exp(logits), num_samples=1, seed=self.seed)
        return action, {}, {}

    def _epsilon_greedy_policy(self, state, **kwargs):
        """Epsilon-greedy policy"""
        eps = self.epsilon()

        # compute probabilities for best action (a* = argmax Q(s,a)), and other actions (a != a*)
        prob_other = tf.cast(eps / self.num_classes, dtype=tf.float32)
        prob_best = 1.0 - eps + prob_other

        q_values = self.dqn.q_values(state, **kwargs)
        best_action = tf.squeeze(tf.argmax(q_values, axis=-1))

        probs = np.full_like(q_values, fill_value=prob_other)
        probs[0][best_action] = prob_best

        # from `probs` obtain `logits` (= log-probs), since tf.random.categorical wants un-normalized probs.
        logits = tf.math.log(probs)
        action = tf.random.categorical(logits=logits, num_samples=1, seed=self.seed)

        return action, {}, dict(prob_best=prob_best, prob_other=prob_other)

    def _epsilon_greedy_policy2(self, state, **kwargs):
        """Epsilon-greedy policy"""
        if self.rng.random() > self.epsilon():
            # greedy action
            q_values = self.dqn.q_values(state, **kwargs)
            action = tf.argmax(q_values, axis=-1)
        else:
            # random action (note: with prob e / |A| is can be still greedy)
            action = self.rng.choice(len(self.num_classes))

        return action, {}, {}

    @tf.function
    def _softmax_policy(self, state, **kwargs):
        """Similar to Boltzmann policy, except that q-values are not exponentiated to get the logits"""
        q_values = self.dqn.q_values(state, **kwargs)
        temperature = self.epsilon()  # 1 => no effect, should decay close to 0 (=> almost greedy selection)

        # normalize `q_values` for numerical stability
        logits = q_values / (temperature + utils.TF_EPS)
        logits = logits - tf.reduce_max(logits)

        action = tf.random.categorical(logits=logits, num_samples=1, seed=self.seed)
        return action, {}, {}

    @tf.function
    def _deterministic_policy(self, state, **kwargs):
        """Deterministic/greedy/argmax policy: always takes action with higher q-value"""
        return self.dqn.act(state, **kwargs), {}, {}

    @tf.function
    def act_randomly(self, state) -> Tuple[tf.Tensor, dict, dict]:
        actions = tf.random.categorical(logits=tf.ones(shape=(1, self.num_classes)), num_samples=1, seed=self.seed)
        return actions, {}, {}

    def learn(self, *args, **kwargs):
        with utils.Timed('Learn'):
            super().learn(*args, **kwargs)

    def update(self):
        if not self.memory.full_enough(amount=self.batch_size):
            return self.memory.update_warning(self.batch_size)

        with utils.Timed('Update', silent=True):
            batch = self.memory.get_batch(batch_size=self.batch_size)

            self.dqn.train_step(batch)
            self.memory.on_update()

    def update_target_network(self):
        if self.should_update_target:
            self.update_target_freq -= 1

            if self.update_target_freq == 0:
                self.dqn.update_target_network(polyak=self.polyak(), copy_weights=True)
                self.update_target_freq = self._update_target_freq
            else:
                self.dqn.update_target_network(polyak=self.polyak(), copy_weights=False)

        self.log(target_weight_distance=self.dqn.debug_target_network())

    def on_transition(self, transition: dict, terminal, exploration=False):
        super().on_transition(transition, terminal, exploration)

        if (self.timestep % self.horizon == 0) or transition['terminal'] or (self.timestep == self.max_timesteps):
            self.memory.end_trajectory()

            if not exploration:
                self.update()
                self.update_target_network()

    # def save_weights(self, path: str):
    #     self.dqn.save_weights(filepath=os.path.join(path, 'dqn'))
    #
    # def load_weights(self):
    #     self.dqn.load_weights(filepath=self.weights_path['dqn'], by_name=False)

    # def summary(self):
    #     self.dqn.summary()


if __name__ == '__main__':
    from rl.parameters import StepDecay, LinearDecay
    from rl.presets import Preset
    from rl.layers.preprocessing import MinMaxScaling
    utils.set_random_seed(42)

    min_max_scaler = MinMaxScaling(min_value=Preset.CARTPOLE_MIN, max_value=Preset.CARTPOLE_MAX)

    # [double+dueling] solved at episode 150
    # agent = DQN(env='CartPole-v1', batch_size=128, policy='e-greedy', clip_norm=None,
    #             epsilon=StepDecay(0.2, steps=100, rate=0.5), lr=1e-3, name='dqn-cart_v1',
    #             dueling=True, prioritized=False, double=True, memory_size=50_000,
    #             gamma=0.99, update_target_network=500, seed=42,
    #             network=dict(units=64, preprocess=dict(state=min_max_scaler)))

    # # [double] solved at 160?
    # agent = DQN(env='CartPole-v1', batch_size=128, policy='e-greedy', clip_norm=None,
    #             epsilon=StepDecay(0.2, steps=100, rate=0.5), lr=1e-3, name='dqn-cart_v1',
    #             dueling=False, prioritized=False, double=True, memory_size=50_000,
    #             gamma=0.99, update_target_network=500, seed=42,
    #             network=dict(units=64, preprocess=dict(state=min_max_scaler)))

    # [DQN] solved at 90
    agent = DQN(env='CartPole-v1', batch_size=128, policy='e-greedy', clip_norm=None,
                epsilon=StepDecay(0.2, steps=100, rate=0.5), lr=1e-3, name='dqn-cart_v1',
                dueling=False, prioritized=False, double=False, memory_size=50_000,
                gamma=0.99, update_target_network=500, seed=42,
                network=dict(units=64, preprocess=dict(state=min_max_scaler)))

    # # DQN + PER: solved at episode 130
    # agent = DQN(env='CartPole-v1', batch_size=128, policy='e-greedy', clip_norm=None,
    #             epsilon=StepDecay(0.2, steps=100, rate=0.5), lr=1e-3, name='dqn_per-cart_v1',
    #             dueling=False, double=False, use_summary=True,
    #             prioritized=True, alpha=0.6, beta=0.1, memory_size=50_000,
    #             gamma=0.99, update_target_network=500, seed=42,
    #             network=dict(units=64, preprocess=dict(state=min_max_scaler)))

    # agent.learn(episodes=200, timesteps=500, save=True, render=False,
    #             evaluation=dict(episodes=20, freq=10), exploration_steps=512)

    agent.load()

    for i in range(20):
        agent.env.seed(42 + i)
        agent.record(timesteps=500)


