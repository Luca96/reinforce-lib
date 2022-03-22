"""Proximal Policy Optimization (PPO)
    - https://arxiv.org/pdf/1707.06347.pdf
"""

import numpy as np
import tensorflow as tf

from rl import utils
from rl.parameters import DynamicParameter

from rl.v2.agents import A2C
from rl.v2.memories import TransitionSpec  # , GAEMemory
from rl.v2.networks import Network, ValueNetwork
from rl.v2.networks.policies import PolicyNetwork

from typing import Dict, Tuple, Union, Callable, List


class ClippedPolicyNetwork(PolicyNetwork):
    """PPO's policy network with clipped objective"""

    @tf.function
    def call(self, inputs, actions=None, **kwargs):
        distribution = Network.call(self, inputs, **kwargs)

        if isinstance(actions, dict) or tf.is_tensor(actions):
            log_prob = distribution.log_prob(actions)
            entropy = distribution.entropy()

            if entropy is None:
                # estimate entropy
                entropy = -tf.reduce_mean(log_prob)

            return log_prob, entropy

        new_actions = tf.identity(distribution)
        return new_actions, distribution.log_prob(new_actions), distribution.mean(), distribution.stddev()

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
            - http://joschu.net/blog/kl-approx.html
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


# class GlobalMemory(GAEMemory):
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#         self.num_actors = self.agent.num_actors  # num. parallel workers in env
#         self.horizon = self.agent.horizon  # max trajectory size
#
#         self.debug_keys = ['returns', 'returns_normalized', 'advantages', 'values',
#                            'advantages_normalized', 'advantages_hist', 'returns_hist']
#
#         # offsets for the storing index (`self.index`) for each actor
#         self.offsets = [0] * self.num_actors
#         self.last_offsets = [0] * self.num_actors  # offsets for GAE computation
#
#     @property
#     def current_size(self) -> int:
#         if self.full:
#             return self.size
#
#         return sum(self.offsets)
#
#     def is_full(self) -> bool:
#         if self.current_size >= self.size:
#             self.full = True
#
#         return self.full
#
#     def full_enough(self, amount: int) -> bool:
#         return self.current_size >= amount
#
#     # def store(self, transition: Dict[str, np.ndarray]):
#     #     assert not self.full
#     #     keys = transition.keys()
#     #
#     #     # unpack `transition` to get a list of per-actor tuples
#     #     for i, experience in enumerate(zip(*transition.values())):
#     #         # set the storing index for the i-th actor
#     #         self.index = self.offsets[i] + (self.horizon * i)
#     #
#     #         # store transition
#     #         super().store(transition={k: v for k, v in zip(keys, experience)})
#     #
#     #         # update index for i-th actor
#     #         self.offsets[i] = self.offsets[i] + 1
#
#     def store(self, transition: Dict[str, np.ndarray]):
#         assert not self.full
#
#         # eventually flatten nested dicts in `transition`; e.g. at "state" and "action"
#         items = list(transition.items())  # is `[(key, np.ndarray or {})]`
#
#         for i, (key, value) in enumerate(items):
#             if isinstance(value, dict):
#                 # flatten dict into `[(key_, value_)]` structure
#                 items[i] = (key, list(value.items()))
#
#         # unpack `items` to get per-actor transitions
#         for i in range(self.num_actors):
#             # set the storing index for the i-th actor
#             self.index = self.offsets[i] + (self.horizon * i)
#
#             # retrieve experience for i-th actor
#             experience = self._index_nested(items, index=i)
#
#             # store transition
#             super().store(transition=experience)
#
#             # update index for i-th actor
#             self.offsets[i] = self.offsets[i] + 1
#
#     def _index_nested(self, items, index: int) -> dict:
#         assert isinstance(items, list) and isinstance(items[0], tuple)
#         out = {}
#
#         for key, value in items:
#             if isinstance(value, list) and isinstance(value[0], tuple):
#                 out[key] = self._index_nested(items=value, index=index)
#             else:
#                 out[key] = value[index]
#
#         return out
#
#     def end_trajectory(self, last_values: list) -> dict:
#         last_values = tf.unstack(last_values)
#         assert len(last_values) == self.num_actors
#
#         data_reward, data_value = self.data['reward'], self.data['value']
#         data_return, data_adv = self.data['return'], self.data['advantage']
#         debug = {k: [] for k in self.debug_keys}
#
#         for i, last_value in enumerate(last_values):
#             stop_idx = self.offsets[i] + (self.horizon * i)
#             start_idx = self.last_offsets[i] + (self.horizon * i)
#
#             # concat value (0 if state is terminal, else bootstrap)
#             value = np.reshape(last_value, newshape=(1, -1))
#             rewards = np.concatenate([data_reward[start_idx:stop_idx], value], axis=0)
#             values = np.concatenate([data_value[start_idx:stop_idx], value], axis=0)
#
#             # compute returns and advantages for i-th actor
#             returns, ret_norm = self.compute_returns(rewards)
#             adv, adv_norm = self.compute_advantages(rewards, values)
#
#             # store them
#             data_return[start_idx:stop_idx] = ret_norm
#             data_adv[start_idx:stop_idx] = adv_norm
#
#             # update index's offsets
#             self.last_offsets[i] = self.offsets[i]
#
#             # update debug
#             dict_ = dict(returns=returns, returns_normalized=ret_norm, advantages=adv, values=values,
#                          advantages_normalized=adv_norm, advantages_hist=adv_norm, returns_hist=ret_norm)
#
#             for k, v in dict_.items():
#                 debug[k].append(v)
#
#         return debug
#
#     def clear(self):
#         super().clear()
#         self.offsets = [0] * self.num_actors
#         self.last_offsets = [0] * self.num_actors
#
#     def get_data(self, **kwargs) -> dict:
#         if self.full:
#             return self.data
#
#         # Get a subset of the data:
#         data = {}
#
#         for key, value in self.data.items():
#             # indexing indices for all the actors
#             indices = []
#
#             for i, offset in enumerate(self.offsets):
#                 start_idx = self.horizon * i
#                 stop_idx = offset + start_idx
#
#                 indices.append(np.arange(start=start_idx, stop=stop_idx))
#
#             indices = np.concatenate(indices)
#             data[key] = value[indices]
#
#         return data
#
#     def to_batches(self, batch_size: int, repeat=1, seed=None) -> tf.data.Dataset:
#         """Returns a tf.data.Dataset iterator over batches of transitions"""
#         assert batch_size >= 1
#         tensors = self.get_data()
#
#         ds = tf.data.Dataset.from_tensor_slices(tensors)
#         ds = ds.shuffle(buffer_size=min(1024, batch_size), reshuffle_each_iteration=True, seed=seed)
#
#         # if size is not a multiple of `batch_size`, just add some data at random
#         size = tensors[list(tensors.keys())[0]].shape[0]
#         remainder = size % batch_size
#
#         if remainder > 0:
#             ds = ds.concatenate(ds.take(count=remainder))
#
#         # batch, repeat, prefetch
#         ds = ds.batch(batch_size)
#         ds = ds.repeat(count=repeat)
#
#         return ds.prefetch(buffer_size=2)


# TODO: option to recompute advantages at each opt epoch/step
class PPO(A2C):
    """PPO agent"""

    def __init__(self, env, horizon: int, batch_size: int, optimization_epochs=10, gamma=0.99,
                 policy_lr: utils.DynamicType = 1e-3, value_lr: utils.DynamicType = 3e-4, optimizer='adam',
                 lambda_=0.95, num_actors=16, name='ppo-agent', clip_ratio: utils.DynamicType = 0.2,
                 policy: dict = None, value: dict = None, entropy: utils.DynamicType = 0.01, clip_norm=(None, None),
                 advantage_scale: utils.DynamicType = 1.0, normalize_advantages: Union[None, str, Callable] = None,
                 normalize_returns: Union[None, str, Callable] = None, target_kl: float = None, target_mse: float = None,
                 clip_ratio_value=None, **kwargs):
        # assert horizon >= 1
        assert optimization_epochs >= 1

        # super().__init__(env, num_actors=num_actors, batch_size=batch_size, gamma=gamma, name=name, **kwargs)

        policy = policy or {}
        policy.setdefault('cls', ClippedPolicyNetwork)

        value = value or {}
        value.setdefault('cls', ClippedValueNetwork)

        super().__init__(env, horizon, num_actors=num_actors, gamma=gamma, name=name, actor_lr=policy_lr,
                         critic_lr=value_lr, actor=policy, critic=value, optimizer=optimizer, clip_norm=clip_norm,
                         entropy=entropy, **kwargs)

        # Hyper-parameters:
        # self.horizon = int(horizon)
        self.opt_epochs = int(optimization_epochs)
        self.lambda_ = tf.constant(lambda_, dtype=tf.float32)
        # self.entropy_strength = DynamicParameter.create(value=entropy)

        self.clip_ratio = DynamicParameter.create(value=clip_ratio)
        self.clip_value = tf.constant(clip_ratio_value or np.inf, dtype=tf.float32)

        # self.adv_scale = DynamicParameter.create(value=advantage_scale)
        # self.adv_normalization_fn = utils.get_normalization_fn(arg=normalize_advantages)
        # self.returns_norm_fn = utils.get_normalization_fn(arg=normalize_returns)

        self.target_kl = tf.constant(target_kl or np.inf, dtype=tf.float32)
        self.target_mse = tf.constant(target_mse or np.inf, dtype=tf.float32)

        # self.policy_lr = DynamicParameter.create(value=policy_lr)
        # self.value_lr = DynamicParameter.create(value=value_lr)

        # Networks
        # self.weights_path = dict(policy=os.path.join(self.base_path, 'policy'),
        #                          value=os.path.join(self.base_path, 'value'))

        # self.policy = Network.create(agent=self, **(policy or {}), base_class=ClippedPolicyNetwork)
        # self.value = Network.create(agent=self, **(value or {}), base_class=ClippedValueNetwork)

        # if clip_norm is None:
        #     clip_norm = (None, None)
        #
        # self.policy.compile(optimizer, clip_norm=clip_norm[0], clip=self.clip_grads, learning_rate=self.policy_lr)
        # self.value.compile(optimizer, clip_norm=clip_norm[1], clip=self.clip_grads, learning_rate=self.value_lr)

        # renaming (note: lr is not renamed)
        self.policy = self.actor
        self.value = self.critic

    @property
    def transition_spec(self) -> TransitionSpec:
        return TransitionSpec(state=self.state_spec, action=(self.num_actions,), next_state=False, terminal=False,
                              reward=(1,), other=dict(log_prob=(self.num_actions,), value=(1,)))

    # def define_memory(self) -> GlobalMemory:
    #     return GlobalMemory(self.transition_spec, agent=self, shape=self.horizon * self.num_actors)

    @property
    def networks(self) -> Dict:
        # redefined networks due to network renaming
        return dict(policy=self.policy, value=self.value)

    @tf.function(jit_compile=True)
    def act(self, states, **kwargs) -> Tuple[tf.Tensor, dict, dict]:
        values = self.value(states, training=False)
        actions, log_prob, mean, std = self.policy(states, training=False)

        other = dict(log_prob=log_prob, value=values)
        debug = dict(distribution_mean=tf.reduce_mean(mean), distribution_std=tf.reduce_mean(std))

        return actions, other, debug

    @tf.function
    def act_evaluation(self, state, **kwargs):
        actions, _, _, _ = self.policy(state, training=False, deterministic=True, **kwargs)
        return actions

    # def learn(self, *args, **kwargs):
    #     with utils.Timed('Learn'):
    #         super().learn(*args, **kwargs)

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

    # def on_transition(self, transition: dict, terminal: np.ndarray, exploration=False):
    #     any_terminal = any(terminal)
    #
    #     if any_terminal:
    #         # adjust terminal states for `MultiprocessingEnv` only
    #         #  - a state is `transition['next_state']` if non-terminal, or
    #         #  - `info['__terminal_state']` is truly terminal
    #         terminal_states = transition['next_state']
    #
    #         for i, info in transition['info'].items():
    #             if '__terminal_state' in info:
    #                 terminal_states[i] = info.pop('__terminal_state')
    #
    #         transition['next_state'] = terminal_states
    #
    #     # TODO: temporary fix
    #     transition.pop('info')
    #
    #     super().on_transition(transition, terminal, exploration)
    #
    #     if any_terminal or (self.timestep % self.horizon == 0) or (self.timestep == self.max_timesteps):
    #         is_failure = tf.reshape(transition['terminal'], shape=(-1, 1))
    #         terminal_states = self.preprocess(transition['next_state'])
    #
    #         values = self.value(terminal_states, training=False)
    #         values = values * utils.to_float(tf.logical_not(is_failure))
    #
    #         debug = self.memory.end_trajectory(last_values=values)
    #         self.log(average=True, **debug)
    #
    #     if not exploration and self.memory.is_full():
    #         self.update()
