"""Advantage Actor-Critic (A2C)"""

import tensorflow as tf
import numpy as np

from rl import utils
from rl.parameters import DynamicParameter

from rl.agents.agents import ParallelAgent
from rl.memories import TransitionSpec, GAEMemory
from rl.networks import Network, ValueNetwork
from rl.networks.policies import PolicyNetwork

from typing import Tuple, Dict


@Network.register(name='A2C-ActorNetwork')
class ActorNetwork(PolicyNetwork):

    def mean(self, distribution: utils.DistributionOrDict) -> utils.TensorOrDict:
        """Returns the average mean of the given `distribution`"""
        if not isinstance(distribution, dict):
            return tf.reduce_mean(distribution.mean())

        return {k: tf.reduce_mean(dist.mean()) for k, dist in distribution.items()}

    def mode(self, distribution: utils.DistributionOrDict) -> utils.TensorOrDict:
        """Returns the average mode of the given `distribution`"""
        if not isinstance(distribution, dict):
            return tf.reduce_mean(distribution.mode())

        return {k: tf.reduce_mean(dist.mode()) for k, dist in distribution.items()}

    def stddev(self, distribution: utils.DistributionOrDict) -> utils.TensorOrDict:
        """Returns the average standard deviation of the given `distribution`"""
        if not isinstance(distribution, dict):
            return tf.reduce_mean(distribution.stddev())

        return {k: tf.reduce_mean(dist.stddev()) for k, dist in distribution.items()}


class ParallelGAEMemory(GAEMemory):
    """GAE memory that support multiple (parallel) environments"""

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
        # size = tensors[list(tensors.keys())[0]].shape[0]
        # remainder = size % batch_size
        remainder = self.index % batch_size

        if remainder > 0:
            ds = ds.concatenate(ds.take(count=remainder))

        # batch, repeat, prefetch
        ds = ds.batch(batch_size)
        ds = ds.repeat(count=repeat)

        return ds.prefetch(buffer_size=2)


class A2C(ParallelAgent):
    """A2C agent"""

    def __init__(self, env, horizon: int, gamma=0.99, name='a2c-agent', optimizer='adam',
                 actor_lr: utils.DynamicType = 1e-3, critic_lr: utils.DynamicType = 3e-4, clip_norm=(None, None),
                 lambda_=1.0, num_actors=16, entropy: utils.DynamicType = 0.01, actor: dict = None, critic: dict = None,
                 advantage_scale: utils.DynamicType = 1.0, advantage_normalization: utils.OptionalStrOrCallable = None,
                 return_normalization: utils.OptionalStrOrCallable = None, **kwargs):
        assert horizon >= 1

        super().__init__(env, num_actors=num_actors, batch_size=horizon * num_actors, gamma=gamma,
                         name=name, **kwargs)

        # Hyper-parameters:
        self.horizon = int(horizon)
        self.lambda_ = tf.constant(lambda_, dtype=tf.float32)
        self.entropy_strength = DynamicParameter.create(value=entropy)

        self.adv_scale = DynamicParameter.create(value=advantage_scale)
        self.adv_normalization_fn = utils.get_normalization_fn(arg=advantage_normalization)
        self.returns_norm_fn = utils.get_normalization_fn(arg=return_normalization)

        self.actor_lr = DynamicParameter.create(value=actor_lr)
        self.critic_lr = DynamicParameter.create(value=critic_lr)

        # Networks
        self.actor = Network.create(agent=self, **(actor or {}), base_class=ActorNetwork)
        self.critic = Network.create(agent=self, **(critic or {}), base_class=ValueNetwork)

        if clip_norm is None:
            clip_norm = (None, None)

        self.actor.compile(optimizer, clip_norm=clip_norm[0], clip=self.clip_grads, learning_rate=self.actor_lr)
        self.critic.compile(optimizer, clip_norm=clip_norm[1], clip=self.clip_grads, learning_rate=self.critic_lr)

    @property
    def transition_spec(self) -> TransitionSpec:
        return TransitionSpec(state=self.state_spec, action=self.action_spec, next_state=False, terminal=False,
                              reward=(1,), other=dict(value=(1,)))

    def define_memory(self) -> ParallelGAEMemory:
        return ParallelGAEMemory(self.transition_spec, agent=self, shape=self.horizon * self.num_actors)

    @tf.function
    def act(self, states, **kwargs) -> Tuple[tf.Tensor, dict, dict]:
        actions, mean, std, mode = self.actor(states, training=False)
        values = self.critic(states, training=False)

        other = dict(value=values)
        debug = dict(distribution_mean=mean, distribution_std=std, distribution_mode=mode)

        return actions, other, debug

    @tf.function
    def act_evaluation(self, state, **kwargs):
        actions, _, _, _ = self.actor(state, training=False, deterministic=True, **kwargs)
        return actions

    def learn(self, *args, **kwargs):
        with utils.Timed('Learn'):
            super().learn(*args, **kwargs)

    def update(self):
        with utils.Timed('Update'):
            batch = self.memory.get_data()

            self.actor.train_step(batch)
            self.critic.train_step(batch)

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
                    state = info.pop('__terminal_state')

                    if isinstance(terminal_states, dict):
                        for k, v in state.items():
                            terminal_states[k][i] = v
                    else:
                        terminal_states[i] = state

            transition['next_state'] = terminal_states

        # TODO: temporary fix
        transition.pop('info')

        super().on_transition(transition, terminal, exploration)

        if any_terminal or (self.timestep % self.horizon == 0) or (self.timestep == self.max_timesteps):
            is_failure = tf.reshape(transition['terminal'], shape=(-1, 1))
            terminal_states = self.preprocess(transition['next_state'])

            values = self.critic(terminal_states, training=False)
            values = values * utils.to_float(tf.logical_not(is_failure))

            debug = self.memory.end_trajectory(last_values=values)
            self.log(average=True, **debug)

        if not exploration and self.memory.is_full():
            self.update()
