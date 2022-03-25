"""Proximal Policy Optimization (PPO)
    - https://arxiv.org/pdf/1707.06347.pdf
"""

import numpy as np
import tensorflow as tf

from rl import utils
from rl.parameters import DynamicParameter

from rl.agents import A2C
from rl.memories import TransitionSpec  # , GAEMemory
from rl.networks import Network, ValueNetwork
# from rl.networks.policies import PolicyNetwork
from rl.agents.a2c import ActorNetwork

from typing import Dict, Tuple, Union, Callable, List


class ClippedPolicyNetwork(ActorNetwork):
    """Policy network with PPO clipped objective"""

    # @tf.function
    # def call(self, inputs, actions=None, **kwargs):
    #     distribution = Network.call(self, inputs, **kwargs)
    #
    #     if isinstance(actions, dict) or tf.is_tensor(actions):
    #         log_prob = distribution.log_prob(actions)
    #         entropy = distribution.entropy()
    #
    #         if entropy is None:
    #             # estimate entropy
    #             entropy = -tf.reduce_mean(log_prob)
    #
    #         return log_prob, entropy
    #
    #     new_actions = tf.identity(distribution)
    #     return new_actions, distribution.log_prob(new_actions), distribution.mean(), distribution.stddev()

    @tf.function
    def call(self, inputs, actions=None, **kwargs):
        distribution: utils.DistributionOrDict = Network.call(self, inputs, **kwargs)

        if isinstance(actions, dict) or tf.is_tensor(actions):
            log_prob = self.log_prob(distribution, actions)
            entropy = self.entropy(distribution)

            if entropy is None:
                # estimate entropy
                entropy = -tf.reduce_mean(log_prob)

            return log_prob, entropy

        new_actions = self.identity(distribution)
        log_probs = self.log_prob(distribution, new_actions)

        return new_actions, log_probs, self.mean(distribution), self.stddev(distribution)

    @tf.function
    def objective(self, batch, reduction=tf.reduce_mean) -> tuple:
        advantages = batch['advantage']
        old_log_prob = batch['log_prob']

        new_log_prob, entropy = self(batch['state'], actions=batch['action'], training=True)

        # KL-Divergence
        kld = self._approx_kl(old_log_prob, new_log_prob)

        # Entropy
        entropy = reduction(entropy)
        entropy_penalty = entropy * self.agent.entropy_strength()

        # Probability ratio
        ratio = tf.math.exp(new_log_prob - old_log_prob)

        # Clipped ratio times advantage
        clip = self.agent.clip_ratio()

        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * tf.clip_by_value(ratio, 1.0 - clip, 1.0 + clip)

        # Loss
        policy_loss = -reduction(tf.minimum(policy_loss_1, policy_loss_2))
        total_loss = policy_loss - entropy_penalty

        # Debug
        clip_fraction = tf.logical_or(ratio < 1.0 - clip, ratio > 1.0 + clip)
        clip_fraction = tf.reduce_mean(tf.cast(clip_fraction, dtype=tf.float32))

        debug = dict(ratio=ratio, log_prob=new_log_prob, old_log_prob=old_log_prob, entropy=entropy, kl_divergence=kld,
                     loss=policy_loss, ratio_clip=clip, loss_entropy=entropy_penalty, loss_total=total_loss,
                     clip_fraction=tf.stop_gradient(clip_fraction))

        return total_loss, debug

    @tf.function
    def _approx_kl(self, old_log_prob, log_prob):
        """Sources:
            - https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py#L247-L253
            - https://joschu.net/blog/kl-approx.html
        """
        log_ratio = log_prob - old_log_prob

        kld = tf.exp(log_ratio - 1.0) - log_ratio
        kld = tf.reduce_mean(kld)

        return tf.stop_gradient(kld)


class ClippedValueNetwork(ValueNetwork):
    """Value network with clipped MSE objective"""

    def structure(self, inputs, name='ClippedValueNetwork', **kwargs) -> tuple:
        return super().structure(inputs, name=name, **kwargs)

    @tf.function
    def objective(self, batch, reduction=tf.reduce_mean) -> tuple:
        states, returns = batch['state'], batch['return']
        old_values = batch['value']

        values = self(states, training=True)
        clipped_values = old_values + tf.clip_by_value(values - old_values,
                                                       -self.agent.clip_value, self.agent.clip_value)
        # compute losses
        mse_loss = tf.square(values - returns)
        clipped_loss = tf.square(clipped_values - returns)

        loss = 0.5 * reduction(tf.maximum(mse_loss, clipped_loss))

        debug = dict(loss=loss, loss_mse=mse_loss, loss_clipped=clipped_loss, clipped=clipped_values,
                     mse=tf.stop_gradient(0.5 * tf.reduce_mean(tf.square(values - old_values))))
        return loss, debug


class PPO(A2C):
    """PPO agent"""

    def __init__(self, env, horizon: int, batch_size: int, optimization_epochs=10, gamma=0.99,
                 policy_lr: utils.DynamicType = 1e-3, value_lr: utils.DynamicType = 3e-4, optimizer='adam',
                 lambda_=0.95, num_actors=16, name='ppo-agent', clip_ratio: utils.DynamicType = 0.2,
                 policy: dict = None, value: dict = None, entropy: utils.DynamicType = 0.01, clip_norm=(None, None),
                 target_kl: float = None, target_mse: float = None, clip_ratio_value=None, **kwargs):
        assert optimization_epochs >= 1
        assert batch_size <= int(horizon * num_actors)

        policy = policy or {}
        policy.setdefault('cls', ClippedPolicyNetwork)

        value = value or {}
        value.setdefault('cls', ClippedValueNetwork)

        super().__init__(env, horizon, num_actors=num_actors, gamma=gamma, name=name, actor_lr=policy_lr,
                         critic_lr=value_lr, actor=policy, critic=value, optimizer=optimizer, clip_norm=clip_norm,
                         entropy=entropy, **kwargs)

        # Hyper-parameters:
        self.opt_epochs = int(optimization_epochs)
        self.lambda_ = tf.constant(lambda_, dtype=tf.float32)
        self.batch_size = int(batch_size)  # redefine batch_size since A2C internally defines it

        self.clip_ratio = DynamicParameter.create(value=clip_ratio)
        self.clip_value = tf.constant(clip_ratio_value or np.inf, dtype=tf.float32)

        self.target_kl = tf.constant(target_kl or np.inf, dtype=tf.float32)
        self.target_mse = tf.constant(target_mse or np.inf, dtype=tf.float32)

        # Networks
        # renaming (note: lr is not renamed)
        self.policy = self.actor
        self.value = self.critic

    @property
    def transition_spec(self) -> TransitionSpec:
        # action=(self.num_actions,)
        num_actions = 1 if isinstance(self.action_spec, dict) else self.action_converter.num_actions

        return TransitionSpec(state=self.state_spec, action=self.action_spec, next_state=False, terminal=False,
                              reward=(1,), other=dict(log_prob=(num_actions,), value=(1,)))

    @property
    def networks(self) -> Dict:
        # redefined networks due to network renaming
        return dict(policy=self.policy, value=self.value)

    @tf.function
    def act(self, states, **kwargs) -> Tuple[tf.Tensor, dict, dict]:
        values = self.value(states, training=False)
        actions, log_prob, mean, std = self.policy(states, training=False)

        other = dict(log_prob=log_prob, value=values)
        debug = dict(distribution_mean=mean, distribution_std=std)

        return actions, other, debug

    @tf.function
    def act_evaluation(self, state, **kwargs):
        actions, _, _, _ = self.policy(state, training=False, deterministic=True, **kwargs)
        return actions

    def update(self):
        if not self.memory.full_enough(amount=self.batch_size):
            return self.memory.update_warning(self.batch_size)

        with utils.Timed('Update'):
            batches = self.memory.to_batches(self.batch_size, repeat=self.opt_epochs, seed=self.seed)
            num_batches = self.memory.current_size // self.batch_size

            train_policy = True
            train_value = True

            for i, batch in enumerate(batches):
                current_epoch = max(1, (i + 1) // num_batches)

                # update policy
                if train_policy:
                    kl = self.policy.train_step(batch, retrieve='kl_divergence')

                    if kl > 1.5 * self.target_kl:
                        train_policy = False

                # update value
                if train_value:
                    mse = self.value.train_step(batch, retrieve='mse')

                    if mse > self.target_mse:
                        train_value = False

                self.log(early_stop_policy=current_epoch if train_policy else 0,
                         early_stop_value=current_epoch if train_value else 0)

                if (not train_policy) and (not train_value):
                    # if both networks were early stopped, terminate updating
                    break

            self.memory.on_update()
            self.memory.clear()
