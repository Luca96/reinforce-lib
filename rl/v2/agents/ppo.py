"""Proximal Policy Optimization (PPO)"""

import os
import gym

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf

from rl import utils
from rl.parameters import DynamicParameter

from rl.v2.agents.agents import ParallelAgent
from rl.v2.memories import TransitionSpec, GAEMemory
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


class GlobalMemory(GAEMemory):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_actors = self.agent.num_actors  # num. parallel workers in env
        self.horizon = self.agent.horizon  # max trajectory size

        self.debug_keys = ['returns', 'returns_normalized', 'advantages', 'values',
                           'advantages_normalized', 'advantages_hist', 'returns_hist']

        # offsets for the storing index (`self.index`) for each actor
        self.offsets = [0] * self.num_actors
        self.last_offsets = [0] * self.num_actors  # offsets for GAE computation

    @property
    def current_size(self) -> int:
        if self.full:
            return self.size

        return sum(self.offsets)

    def is_full(self) -> bool:
        if self.current_size >= self.size:
            self.full = True

        return self.full

    def full_enough(self, amount: int) -> bool:
        return self.current_size >= amount

    # def store(self, transition: Dict[str, np.ndarray]):
    #     assert not self.full
    #     keys = transition.keys()
    #
    #     # unpack `transition` to get a list of per-actor tuples
    #     for i, experience in enumerate(zip(*transition.values())):
    #         # set the storing index for the i-th actor
    #         self.index = self.offsets[i] + (self.horizon * i)
    #
    #         # store transition
    #         super().store(transition={k: v for k, v in zip(keys, experience)})
    #
    #         # update index for i-th actor
    #         self.offsets[i] = self.offsets[i] + 1

    def store(self, transition: Dict[str, np.ndarray]):
        assert not self.full

        # eventually flatten nested dicts in `transition`; e.g. at "state" and "action"
        items = list(transition.items())  # is `[(key, np.ndarray or {})]`

        for i, (key, value) in enumerate(items):
            if isinstance(value, dict):
                # flatten dict into `[(key_, value_)]` structure
                items[i] = (key, list(value.items()))

        # unpack `items` to get per-actor transitions
        for i in range(self.num_actors):
            # set the storing index for the i-th actor
            self.index = self.offsets[i] + (self.horizon * i)

            # retrieve experience for i-th actor
            experience = self._index_nested(items, index=i)

            # store transition
            super().store(transition=experience)

            # update index for i-th actor
            self.offsets[i] = self.offsets[i] + 1

    def _index_nested(self, items, index: int) -> dict:
        assert isinstance(items, list) and isinstance(items[0], tuple)
        out = {}

        for key, value in items:
            if isinstance(value, list) and isinstance(value[0], tuple):
                out[key] = self._index_nested(items=value, index=index)
            else:
                out[key] = value[index]

        return out

    def end_trajectory(self, last_values: list) -> dict:
        last_values = tf.unstack(last_values)
        assert len(last_values) == self.num_actors

        data_reward, data_value = self.data['reward'], self.data['value']
        data_return, data_adv = self.data['return'], self.data['advantage']
        debug = {k: [] for k in self.debug_keys}

        for i, last_value in enumerate(last_values):
            stop_idx = self.offsets[i] + (self.horizon * i)
            start_idx = self.last_offsets[i] + (self.horizon * i)

            # concat value (0 if state is terminal, else bootstrap)
            value = np.reshape(last_value, newshape=(1, -1))
            rewards = np.concatenate([data_reward[start_idx:stop_idx], value], axis=0)
            values = np.concatenate([data_value[start_idx:stop_idx], value], axis=0)

            # compute returns and advantages for i-th actor
            returns, ret_norm = self.compute_returns(rewards)
            adv, adv_norm = self.compute_advantages(rewards, values)

            # store them
            data_return[start_idx:stop_idx] = ret_norm
            data_adv[start_idx:stop_idx] = adv_norm

            # update index's offsets
            self.last_offsets[i] = self.offsets[i]

            # update debug
            dict_ = dict(returns=returns, returns_normalized=ret_norm, advantages=adv, values=values,
                         advantages_normalized=adv_norm, advantages_hist=adv_norm, returns_hist=ret_norm)

            for k, v in dict_.items():
                debug[k].append(v)

        return debug

    def clear(self):
        super().clear()
        self.offsets = [0] * self.num_actors
        self.last_offsets = [0] * self.num_actors

    def get_data(self, **kwargs) -> dict:
        if self.full:
            return self.data

        # Get a subset of the data:
        data = {}

        for key, value in self.data.items():
            # indexing indices for all the actors
            indices = []

            for i, offset in enumerate(self.offsets):
                start_idx = self.horizon * i
                stop_idx = offset + start_idx

                indices.append(np.arange(start=start_idx, stop=stop_idx))

            indices = np.concatenate(indices)
            data[key] = value[indices]

        return data

    def to_batches(self, batch_size: int, repeat=1, seed=None) -> tf.data.Dataset:
        """Returns a tf.data.Dataset iterator over batches of transitions"""
        assert batch_size >= 1
        tensors = self.get_data()

        ds = tf.data.Dataset.from_tensor_slices(tensors)
        ds = ds.shuffle(buffer_size=min(1024, batch_size), reshuffle_each_iteration=True, seed=seed)

        # if size is not a multiple of `batch_size`, just add some data at random
        size = tensors[list(tensors.keys())[0]].shape[0]
        remainder = size % batch_size

        if remainder > 0:
            ds = ds.concatenate(ds.take(count=remainder))

        # batch, repeat, prefetch
        ds = ds.batch(batch_size)
        ds = ds.repeat(count=repeat)

        return ds.prefetch(buffer_size=2)


# TODO: option to recompute advantages at each opt epoch/step
# TODO: inherit from `A2C`
class PPO(ParallelAgent):
    """PPO agent"""

    def __init__(self, env, horizon: int, batch_size: int, optimization_epochs=10, gamma=0.99,
                 policy_lr: utils.DynamicType = 1e-3, value_lr: utils.DynamicType = 3e-4, optimizer='adam',
                 lambda_=0.95, num_actors=16, name='ppo-agent', clip_ratio: utils.DynamicType = 0.2,
                 policy: dict = None, value: dict = None, entropy: utils.DynamicType = 0.01, clip_norm=(None, None),
                 advantage_scale: utils.DynamicType = 1.0, normalize_advantages: Union[None, str, Callable] = None,
                 normalize_returns: Union[None, str, Callable] = None, target_kl: float = None, target_mse: float = None,
                 clip_ratio_value=None, **kwargs):
        assert horizon >= 1
        assert optimization_epochs >= 1

        super().__init__(env, num_actors=num_actors, batch_size=batch_size, gamma=gamma, name=name, **kwargs)

        # Hyper-parameters:
        self.horizon = int(horizon)
        self.opt_epochs = int(optimization_epochs)
        self.lambda_ = tf.constant(lambda_, dtype=tf.float32)
        self.entropy_strength = DynamicParameter.create(value=entropy)

        self.clip_ratio = DynamicParameter.create(value=clip_ratio)
        self.clip_value = tf.constant(clip_ratio_value or np.inf, dtype=tf.float32)

        self.adv_scale = DynamicParameter.create(value=advantage_scale)
        self.adv_normalization_fn = utils.get_normalization_fn(arg=normalize_advantages)
        self.returns_norm_fn = utils.get_normalization_fn(arg=normalize_returns)

        self.target_kl = tf.constant(target_kl or np.inf, dtype=tf.float32)
        self.target_mse = tf.constant(target_mse or np.inf, dtype=tf.float32)

        self.policy_lr = DynamicParameter.create(value=policy_lr)
        self.value_lr = DynamicParameter.create(value=value_lr)

        # Networks
        self.weights_path = dict(policy=os.path.join(self.base_path, 'policy'),
                                 value=os.path.join(self.base_path, 'value'))

        self.policy = Network.create(agent=self, **(policy or {}), base_class=ClippedPolicyNetwork)
        self.value = Network.create(agent=self, **(value or {}), base_class=ClippedValueNetwork)

        if clip_norm is None:
            clip_norm = (None, None)

        self.policy.compile(optimizer, clip_norm=clip_norm[0], clip=self.clip_grads, learning_rate=self.policy_lr)
        self.value.compile(optimizer, clip_norm=clip_norm[1], clip=self.clip_grads, learning_rate=self.value_lr)

    @property
    def transition_spec(self) -> TransitionSpec:
        return TransitionSpec(state=self.state_spec, action=(self.num_actions,), next_state=False, terminal=False,
                              reward=(1,), other=dict(log_prob=(self.num_actions,), value=(1,)))

    def define_memory(self) -> GlobalMemory:
        return GlobalMemory(self.transition_spec, agent=self, shape=self.horizon * self.num_actors)

    @tf.function(jit_compile=True)
    def act(self, states, **kwargs) -> Tuple[tf.Tensor, dict, dict]:
        # states = self.stack_states(states)

        values = self.value(states, training=False)
        actions, log_prob, mean, std = self.policy(states, training=False)

        other = dict(log_prob=log_prob, value=values)
        debug = dict(distribution_mean=tf.reduce_mean(mean), distribution_std=tf.reduce_mean(std))

        return actions, other, debug

    @tf.function
    def act_evaluation(self, state, **kwargs):
        actions, _, _, _ = self.policy(state, training=False, deterministic=True, **kwargs)
        return actions

    # def learn(self, episodes: int, timesteps: int, render: Union[bool, int, None] = False, should_close=True,
    #           evaluation: Union[dict, bool] = None, exploration_steps=0, save=True):
    #     assert episodes > 0
    #     assert timesteps > 0
    #
    #     import time
    #     t0 = time.time()
    #     total_seconds = 0
    #
    #     self.on_start(episodes, timesteps)
    #
    #     # init evaluation args:
    #     if isinstance(evaluation, dict):
    #         eval_freq = evaluation.pop('freq', episodes + 1)  # default: never evaluate
    #         assert isinstance(eval_freq, int)
    #
    #         evaluation['should_close'] = False
    #         evaluation.setdefault('episodes', 1)  # default: evaluate on just 1 episode
    #         evaluation.setdefault('timesteps', timesteps)  # default: evaluate on the same number of timesteps
    #         evaluation.setdefault('render', render)  # default: same rendering options
    #     else:
    #         evaluation = {}
    #         eval_freq = episodes + 1  # never evaluate
    #
    #     for episode in range(episodes):
    #         self.on_episode_start(episode)
    #
    #         episode_reward = 0.0
    #         discounted_reward = 0.0
    #         discount = 1.0
    #         ti = time.time()
    #
    #         states = self.env.reset()
    #         states = self.preprocess(states)
    #
    #         # inner-loop:
    #         t = 0
    #
    #         while t < timesteps + 1:
    #             t += 1
    #             self.timestep = t
    #             self.total_steps += 1
    #
    #             # Agent prediction
    #             actions, other, debug = self.act(states)
    #             actions_env = self.convert_action(actions)
    #
    #             # Environment step
    #             next_states, rewards, terminals, info = self.env.step(action=actions_env)
    #
    #             is_truncated = [x.get('TimeLimit.truncated', False) for x in info.values()]
    #             is_failure = np.logical_and(terminals, np.logical_not(is_truncated))
    #
    #             episode_reward += np.mean(rewards)
    #             discounted_reward += np.mean(rewards) * discount
    #             discount *= self.gamma
    #
    #             transition = dict(state=states, action=actions, reward=rewards, next_state=next_states,
    #                               terminal=is_failure, info=info, **(other or {}))
    #
    #             self.on_transition(transition, terminals)
    #             self.log_env(action=actions_env, **debug)
    #
    #             if (t == timesteps) or self.memory.is_full():
    #                 seconds = time.time() - ti
    #                 total_seconds += seconds
    #
    #                 print(f'Episode {episode} terminated after {t} timesteps in {round(seconds, 3)}s ' +
    #                       f'with reward {round(episode_reward, 3)}')
    #
    #                 self.log(timestep=t, total_steps=self.total_steps, time_seconds=total_seconds,
    #                          seconds_per_epoch=seconds)
    #                 self.on_termination(last_transition=transition)
    #                 break
    #             else:
    #                 # if any(terminals):
    #                 #     reset_states = self.env.reset(terminating=terminals)
    #                 #
    #                 #     for idx, state in reset_states:
    #                 #         next_states[idx] = state
    #
    #                 states = next_states
    #                 states = self.preprocess(states)
    #
    #         self.on_episode_end(episode, episode_reward)
    #
    #         # Evaluate
    #         if episode % eval_freq == 0:
    #             eval_rewards = self.evaluate2(**evaluation)
    #             self.log(eval_rewards=eval_rewards)
    #
    #             print(f'[Evaluation] average return: {np.round(np.mean(eval_rewards), 2)}, '
    #                   f'std: {np.round(np.std(eval_rewards), 2)}')
    #
    #     print(f'Time taken {round(time.time() - t0, 3)}s.')
    #     self.on_close(should_close)

    def learn(self, *args, **kwargs):
        with utils.Timed('Learn'):
            super().learn(*args, **kwargs)

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
                # self.log(early_stop_epoch=current_epoch, early_stop_i=i,
                #          early_stop_num_batches=num_batches)

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

    def on_transition(self, transition: dict, terminal: np.ndarray, exploration=False):
        any_terminal = any(terminal)

        if any_terminal:
            # adjust terminal states for `MultiprocessingEnv` only
            #  - a state is `transition['next_state']` if non-terminal, or
            #  - `info['__terminal_state']` is truly terminal
            terminal_states = transition['next_state']

            for i, info in transition['info'].items():
                if '__terminal_state' in info:
                    terminal_states[i] = info.pop('__terminal_state')

            transition['next_state'] = terminal_states

        # TODO: temporary fix
        transition.pop('info')

        super().on_transition(transition, terminal, exploration)

        if any_terminal or (self.timestep % self.horizon == 0) or (self.timestep == self.max_timesteps):
            is_failure = tf.reshape(transition['terminal'], shape=(-1, 1))
            terminal_states = self.preprocess(transition['next_state'])

            values = self.value(terminal_states, training=False)
            values = values * utils.to_float(tf.logical_not(is_failure))

            debug = self.memory.end_trajectory(last_values=values)
            self.log(average=True, **debug)

        if not exploration and self.memory.is_full():
            self.update()

    # def load_weights(self):
    #     self.policy.load_weights(filepath=self.weights_path['policy'], by_name=False)
    #     self.value.load_weights(filepath=self.weights_path['value'], by_name=False)
    #
    # def save_weights(self):
    #     self.policy.save_weights(filepath=self.weights_path['policy'])
    #     self.value.save_weights(filepath=self.weights_path['value'])
    #
    # def summary(self):
    #     self.policy.summary()
    #     self.value.summary()


if __name__ == '__main__':
    from rl.environments.gym.parallel import MultiProcessEnv, SequentialEnv
    utils.set_random_seed(42)
    # utils.tf_enable_debug()

    from rl.environments.debug import DictActionEnv

    # env = DictActionEnv()
    # s = env.reset()
    # a = env.action_space.sample()
    # x = env.step(a)
    # breakpoint()

    class CompoundPolicyNetwork(ClippedPolicyNetwork):

        @tf.function
        def call(self, inputs, actions=None, **kwargs):
            distributions = Network.call(self, inputs, **kwargs)

            if isinstance(actions, dict) or tf.is_tensor(actions):
                log_prob = self.log_prob(distributions, actions)
                entropy = self.entropy(distributions)

                if entropy is None:
                    # estimate entropy
                    entropy = -tf.reduce_mean(log_prob)

                return log_prob, entropy

            new_actions = self.identity(distributions)

            return new_actions, self.log_prob(distributions, new_actions), \
                   self.mean(distributions), self.stddev(distributions)

        def output_layer(self, layer, **kwargs) -> tf.keras.layers.Layer:
            # TODO: missing kwargs
            from rl.layers.distributions import DistributionLayer
            distributions = DistributionLayer.get(action_space=self.agent.env.action_space)
            return {k: distribution(layer) for k, distribution in distributions.items()}

        def identity(self, distributions: dict):
            return {k: tf.identity(v) for k, v in distributions.items()}

        def log_prob(self, distributions: dict, x: dict):
            log_probs = [distribution.log_prob(x[k]) for k, distribution in distributions.items()]
            log_probs = tf.concat(log_probs, axis=-1)

            # assume distributions to be independent
            return tf.reduce_sum(log_probs, axis=-1, keepdims=True)

        def entropy(self, distributions: dict):
            entropies = [distributions.entropy() for distributions in distributions.values()]

            if any(x is None for x in entropies):
                return None

            entropies = tf.concat(entropies, axis=-1)

            # https://en.wikipedia.org/wiki/Joint_entropy#Less_than_or_equal_to_the_sum_of_individual_entropies
            return tf.reduce_sum(entropies, axis=-1, keepdims=True)

        def mean(self, distributions: dict) -> tf.Tensor:
            # return tf.concat([distribution.mean() for distribution in distributions.values()], -1)
            return 0.0

        def stddev(self, distributions: dict) -> tf.Tensor:
            # return tf.concat([distribution.stddev() for distribution in distributions.values()], -1)
            return 0.0


    class MyPPO(PPO):

        @property
        def transition_spec(self) -> TransitionSpec:
            return TransitionSpec(state=self.state_spec, action=self.action_spec, next_state=False, terminal=False,
                                  reward=(1,), other=dict(log_prob=(self.num_actions,), value=(1,)))

        def log_transition(self, transition: Dict[str, list]):
            self.log(reward=np.mean(transition['reward']), action=transition['action'])

        def _init_action_space(self):
            discrete_space = self.env.action_space['discrete']
            continuous_space = self.env.action_space['continuous']

            self.num_actions = 1
            self.num_classes = discrete_space.n

            self.action_low = tf.constant(continuous_space.low, dtype=tf.float32)
            self.action_high = tf.constant(continuous_space.high, dtype=tf.float32)
            self.action_range = tf.constant(continuous_space.high - continuous_space.low, dtype=tf.float32)

            def convert_action(actions: dict) -> List[dict]:
                discrete = actions['action_discrete']
                continuous = actions['action_continuous']

                out = []

                for d, c in zip(tf.unstack(discrete), tf.unstack(continuous)):
                    d = tf.cast(tf.squeeze(d), dtype=tf.int32)
                    c = tf.reshape(c * self.action_range + self.action_low, shape=(1,))

                    out.append(dict(discrete=d.numpy(), continuous=c.numpy()))

                return out

            self.convert_action = convert_action

    agent = MyPPO(env=DictActionEnv, horizon=16, batch_size=64, optimization_epochs=2,
                  policy=dict(cls=CompoundPolicyNetwork, units=[16, 32]),
                  num_actors=4, seed=utils.GLOBAL_SEED, use_summary=False)

    # agent.summary()
    # agent.memory.summary()
    # breakpoint()

    agent.learn(episodes=100, timesteps=100)
    exit()

    # print('CUDA:', tf.test.is_gpu_available(), tf.test.is_built_with_cuda())
    # print('logical:\n', tf.config.list_logical_devices())
    # print('physical:\n', tf.config.list_physical_devices())
    # exit()

    # solved at epoch 50
    # agent = PPO(env='CartPole-v1', name='ppo-cart_v1', horizon=64, batch_size=256,
    #             optimization_epochs=10, policy_lr=3e-4, num_actors=16,
    #             entropy=1e-3, clip_norm=(5.0, 5.0), use_summary=True,
    #             policy=dict(units=32), value=dict(units=64),
    #             target_kl=0.3, target_mse=1.0, parallel_env=SequentialEnv,
    #             seed=utils.GLOBAL_SEED)
    #
    # agent.learn(episodes=100, timesteps=500, save=True, should_close=True,
    #             evaluation=dict(episodes=25, freq=10))

    # max +50 reward
    # agent = PPO(env='LunarLander-v2', name='ppo-lunar', horizon=128, batch_size=512,
    #             optimization_epochs=5, policy_lr=3e-4, num_actors=16,
    #             entropy=1e-4, clip_norm=(5.0, 5.0), use_summary=True,
    #             policy=dict(units=128, activation='tanh'), value=dict(units=256),
    #             target_kl=0.3, target_mse=1.0, parallel_env=SequentialEnv,
    #             # env_kwargs={'processes': 4},
    #             seed=utils.GLOBAL_SEED)

    # try normalization, reward, scaling, ...
    # agent = PPO(env='LunarLander-v2', name='ppo-lunar', horizon=128, batch_size=512,
    #             optimization_epochs=5, policy_lr=1e-3, num_actors=16,
    #             # decay entropy or 0?
    #             entropy=0,
    #             clip_norm=(None, 5.0), use_summary=False,
    #             policy=dict(units=128, activation=tf.nn.swish), value=dict(units=128),
    #             target_kl=0.3, target_mse=1.0, parallel_env=SequentialEnv,
    #             seed=utils.GLOBAL_SEED)

    # achieves 60+ reward
    # agent = PPO(env='LunarLander-v2', name='ppo-lunar', horizon=250, batch_size=400,
    #             optimization_epochs=10, policy_lr=3e-4, num_actors=16,
    #             entropy=1e-3, clip_norm=(5.0, 5.0), use_summary=True,
    #             policy=dict(units=64), value=dict(units=128),
    #             target_kl=0.3, target_mse=1.0, parallel_env=SequentialEnv,
    #             seed=utils.GLOBAL_SEED)

    from rl.parameters import StepDecay
    agent = PPO(env='LunarLander-v2', name='ppo-lunar', horizon=250-50, batch_size=320,
                optimization_epochs=10, policy_lr=3e-4, num_actors=16,
                entropy=StepDecay(1e-3, steps=50, rate=0.5), clip_norm=(5.0, 5.0),
                use_summary=True,
                policy=dict(units=64), value=dict(units=128),
                target_kl=0.35, target_mse=1.0, parallel_env=SequentialEnv,
                seed=utils.GLOBAL_SEED)

    agent.learn(episodes=250, timesteps=250-50, save=True, should_close=True,
                evaluation=dict(episodes=25, freq=10))
    exit()

    # pendulum (need more training, at lest 50/100 episodes more)
    agent = PPO(env='Pendulum-v1', name='ppo-pendulum', horizon=200, batch_size=256,
                optimization_epochs=5, policy_lr=1e-3, value_lr=1e-3, num_actors=8,
                entropy=1e-3, use_summary=True,
                policy=dict(units=128), value=dict(units=128),
                parallel_env=SequentialEnv, seed=utils.GLOBAL_SEED)

    # > 110 eps reward; 150 eps; 200 timesteps
    # agent = PPO(env='LunarLanderContinuous-v2', name='ppo-lunar_c', horizon=100, batch_size=256,
    #             optimization_epochs=5, policy_lr=1e-3, value_lr=1e-3, num_actors=16,
    #             entropy=1e-3, use_summary=True,
    #             policy=dict(units=128), value=dict(units=128),
    #             parallel_env=SequentialEnv, seed=utils.GLOBAL_SEED)

    def convert_action():
        space = agent.env.action_space
        assert isinstance(space, gym.spaces.Box) and space.is_bounded()

        low = tf.constant(space.low, dtype=tf.float32)
        delta = tf.constant(space.high - space.low, dtype=tf.float32)

        # return lambda actions: [np.squeeze(a * delta + low) for a in actions]  # lunar-c
        return lambda actions: [np.reshape(a * delta + low, newshape=1) for a in actions]  # pendulum
        # return lambda actions: [np.reshape((a + 1.0) / 2.0 * delta + low, newshape=1) for a in actions]

    agent.convert_action = convert_action()

    # agent.summary()
    # breakpoint()

    agent.learn(episodes=150+50, timesteps=200, save=False, should_close=True,
                evaluation=dict(episodes=25, timesteps=200, freq=5, render=False))
