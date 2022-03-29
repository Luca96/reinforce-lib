"""Vanilla Policy Gradient (VPG) Algorithm"""

import numpy as np
import tensorflow as tf

from rl import utils
from rl.parameters import DynamicParameter

from rl.agents import Agent
from rl.memories import EpisodicMemory, TransitionSpec
from rl.networks import Network, ValueNetwork
from rl.networks.policies import PolicyNetwork

from typing import Tuple, Dict


class DiscountedMemory(EpisodicMemory):
    def __init__(self, *args, agent, **kwargs):
        super().__init__(*args, **kwargs)
        self.assert_reserved(keys=['return', 'advantage', 'discount'])

        self.data['return'] = np.zeros_like(self.data['value'])
        self.data['advantage'] = np.zeros(shape=self.shape + (1,), dtype=np.float32)
        self.data['discount'] = np.zeros_like(self.data['value'])
        self.agent = agent

    def end_trajectory(self, last_value) -> dict:
        data_reward, data_value = self.data['reward'], self.data['value']
        data_return, data_adv = self.data['return'], self.data['advantage']

        v = np.reshape(last_value, newshape=(1, -1))
        rewards = np.concatenate([data_reward[:self.index], v], axis=0)
        values = data_value[:self.index]

        # compute returns and advantages
        returns = utils.rewards_to_go(rewards, discount=self.agent.gamma)
        advantages = returns.reshape(values.shape) - values

        # store them
        data_return[:self.index] = returns
        data_adv[:self.index] = advantages

        discounts = np.logspace(0, self.index, num=self.index, base=self.agent.gamma, endpoint=False)
        discounts = discounts.reshape(-1, 1)
        self.data['discount'][:self.index] = discounts

        return dict(returns=returns, advantages=advantages, values=values, discounts=discounts)


class DiscountedPolicyNetwork(PolicyNetwork):

    @tf.function
    def objective(self, batch, reduction=tf.reduce_mean) -> tuple:
        advantages = batch['advantage']
        log_prob, entropy = self(batch['state'], actions=batch['action'], training=True)

        # Entropy
        entropy = reduction(entropy)
        entropy_loss = entropy * self.agent.entropy_strength()

        # Loss (discounted PG)
        policy_loss = -reduction(batch['discount'] * log_prob * advantages)
        total_loss = policy_loss - entropy_loss

        # Debug
        debug = dict(log_prob=log_prob, entropy=entropy, loss=policy_loss, loss_entropy=entropy_loss,
                     loss_total=total_loss)

        return total_loss, debug


# TODO: support for recurrent policy/value?
class VPG(Agent):
    """VPG implementation as detailed by GRL book (page 354)"""

    def __init__(self, env, horizon: int, gamma=0.99, name='vpg-agent', policy_lr: utils.DynamicType = 3e-4,
                 value_lr: utils.DynamicType = 1e-3, policy: dict = None, value: dict = None,
                 entropy: utils.DynamicType = 1e-3, optimizer='adam', clip_norm=(None, None), **kwargs):
        super().__init__(env, batch_size=int(horizon), gamma=gamma, name=name, **kwargs)

        # Hyper-parameters:
        self.horizon = int(horizon)
        self.entropy_strength = DynamicParameter.create(value=entropy)

        self.policy_lr = DynamicParameter.create(value=policy_lr)
        self.value_lr = DynamicParameter.create(value=value_lr)

        if clip_norm is None:
            clip_norm = (None, None)

        # Networks
        self.policy = Network.create(agent=self, **(policy or {}), base_class=DiscountedPolicyNetwork)
        self.value = Network.create(agent=self, **(value or {}), base_class=ValueNetwork)

        self.policy.compile(optimizer, clip_norm=clip_norm[0], clip=self.clip_grads, learning_rate=self.policy_lr)
        self.value.compile(optimizer, clip_norm=clip_norm[1], clip=self.clip_grads, learning_rate=self.value_lr)

    @property
    def transition_spec(self) -> TransitionSpec:
        return TransitionSpec(state=self.state_spec, action=self.action_spec, next_state=False, terminal=False,
                              reward=(1,), other=dict(value=(1,)))

    def define_memory(self) -> DiscountedMemory:
        return DiscountedMemory(self.transition_spec, agent=self, shape=(self.horizon,), seed=self.seed)

    @tf.function
    def act(self, state, **kwargs) -> Tuple[tf.Tensor, dict, dict]:
        action, mean, std, mode = self.policy(state, training=False, **kwargs)
        value = self.value(state, training=False, **kwargs)

        other = dict(value=value)
        debug = dict(distribution_mean=mean, distribution_std=std, distribution_mode=mode)

        return action, other, debug

    @tf.function
    def act_randomly(self, state) -> Tuple[tf.Tensor, dict, dict]:
        action, _, _, _ = self.policy(state, training=False)
        return action, {}, {}

    def update(self):
        assert self.memory.size >= 1

        with utils.Timed('Update'):
            batch = self.memory.get_data()

            # update networks
            self.policy.train_step(batch)
            self.value.train_step(batch)

        self.memory.clear()

    def learn(self, *args, **kwargs):
        with utils.Timed('Learn'):
            super().learn(*args, **kwargs)

    def on_transition(self, transition: Dict[str, list], terminal: bool, exploration=False):
        super().on_transition(transition, terminal, exploration)

        if terminal or (self.timestep == self.max_timesteps) or self.memory.is_full():
            terminal_state = self.preprocess(transition['next_state'])

            value = self.value(terminal_state, training=False)
            value = value * utils.to_float(tf.logical_not(transition['terminal']))

            debug = self.memory.end_trajectory(last_value=value)
            self.log(average=True, **debug)

            if not exploration:
                self.update()

    def log_transition(self, transition: dict):
        super().log_transition(transition)

        self.log(action_hist=transition['action'], reward_hist=transition['reward'], state_hist=transition['state'],
                 value_hist=transition['value'])
