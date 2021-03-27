"""Synchronous Advantage Actor-Critic (A2C)"""

import os
import time
import numpy as np
import multiprocessing as mp
import tensorflow as tf

from rl import utils
from rl.parameters import DynamicParameter

from rl.v2.agents import Agent
from rl.v2.memories import TransitionSpec, EpisodicMemory
from rl.v2.networks import Network, PolicyNetwork, DecomposedValueNetwork

from typing import Tuple, Union


# TODO: `exploration` and `evaluation` only on A3C not workers
# TODO: loading and saving
# TODO: memory intensive (N+1 copies of networks' weights)
class A3C(Agent):
    def __init__(self, env, name='a3c-agent', parallel_actors=None, n_steps=5, load=False, entropy=0.01,
                 optimizer='rmsprop', lambda_=1.0, actor: dict = None, wait_time=0.001, gamma=0.99,
                 normalize_advantages: Union[None, str] = 'sign', critic: dict = None,
                 advantage_scale: utils.DynamicType = 1.0, actor_lr: utils.DynamicType = 7e-4,
                 clip_norm: Tuple[utils.DynamicType] = (1.0, 1.0), critic_lr: utils.DynamicType = 7e-4, **kwargs):
        assert n_steps >= 1
        assert wait_time >= 0.0
        assert callable(env) or isinstance(env, str)

        self.n_steps = int(n_steps)
        self.wait_time = float(wait_time)

        super().__init__(env, batch_size=n_steps, gamma=gamma, name=name, **kwargs)
        self.super = super()
        kwargs.pop('use_summary', None)

        if parallel_actors is None:
            parallel_actors = mp.cpu_count()
        else:
            assert parallel_actors >= 1

        self.num_actors = int(parallel_actors)

        # Hyper-parameters
        self.lambda_ = tf.constant(lambda_, dtype=tf.float32)
        self.entropy_strength = DynamicParameter.create(value=entropy)
        self.adv_scale = DynamicParameter.create(value=advantage_scale)
        self.adv_normalization_fn = utils.get_normalization_fn(arg=normalize_advantages)

        self.actor_lr = DynamicParameter.create(value=actor_lr)
        self.critic_lr = DynamicParameter.create(value=critic_lr)

        # Networks
        self.actor_args = actor if isinstance(actor, dict) else {}
        self.critic_args = critic if isinstance(critic, dict) else {}
        self.optimizer_args = optimizer if isinstance(optimizer, dict) else dict(name='rmsprop')
        self.clip_norm = clip_norm

        self.actor: ActorNetwork = None
        self.critic: CriticNetwork = None

        self.weights_path = dict(policy=os.path.join(self.base_path, 'actor'),
                                 value=os.path.join(self.base_path, 'critic'))

        # Queues (shared memory across workers)
        self.params_queue = mp.Queue()
        self.grads_queue = mp.Queue()

        # Workers
        self.workers = None
        self.worker_args = dict(env=env, n_steps=self.n_steps, load=False, entropy=entropy, optimizer=optimizer,
                                lambda_=lambda_, actor=self.actor_args, normalize_advantages=normalize_advantages,
                                critic=self.critic_args, advantage_scale=advantage_scale, actor_lr=actor_lr,
                                clip_norm=clip_norm, critic_lr=critic_lr, params_queue=self.params_queue,
                                grads_queue=self.grads_queue,
                                summary_queue=self.summary_queue if self.should_log else None, **kwargs)
        if load:
            self.load()

    @property
    def transition_spec(self) -> TransitionSpec:
        return TransitionSpec(state=self.state_spec, action=(self.num_actions,), next_state=False, terminal=False,
                              other=dict(value=(2,)))

    @property
    def memory(self) -> 'GAEMemory':
        if self._memory is None:
            self._memory = GAEMemory(self.transition_spec, agent=self, shape=self.n_steps)

        return self._memory

    def _init_networks(self):
        if self.actor is not None:
            return

        self.actor = Network.create(agent=self, log_prefix=f'actor', **self.actor_args, base_class='A3C-ActorNetwork')
        self.critic = Network.create(agent=self, log_prefix=f'critic', **self.critic_args,
                                     base_class='A3C-CriticNetwork')

        # self.actor = ActorNetwork(agent=self, log_prefix=f'actor', **self.actor_args)
        # self.critic = CriticNetwork(agent=self, log_prefix=f'critic', **self.critic_args)

        optimizer = self.optimizer_args.pop('name', 'rmsprop')
        self.actor.compile(optimizer, clip_norm=self.clip_norm[0], learning_rate=self.actor_lr, **self.optimizer_args)
        self.critic.compile(optimizer, clip_norm=self.clip_norm[1], learning_rate=self.critic_lr, **self.optimizer_args)

    def _init_workers(self, *args, **kwargs):
        self.workers = [Worker(idx=i + 1, learn_args=args, learn_kwargs=kwargs, **self.worker_args)
                        for i in range(self.num_actors)]

    @tf.function
    def act(self, state) -> Tuple[tf.Tensor, dict, dict]:
        action, _, mean, std = self.actor(state, training=False)
        value = self.critic(state, training=False)

        other = dict(value=value)

        if self.distribution_type != 'categorical':
            debug = dict(distribution_mean=mean, distribution_std=std)
        else:
            debug = dict()

        return action, other, debug

    def update_and_send(self):
        while any([actor.is_alive() for actor in self.workers]):
            # wait for gradients
            while self.grads_queue.empty():
                time.sleep(self.wait_time)

            # consume gradients
            consumed_grads = 0

            while not self.grads_queue.empty():
                actor_grads, critic_grads = self.grads_queue.get()

                # apply gradients
                self.actor.update(gradients=actor_grads)
                self.critic.update(gradients=critic_grads)

                consumed_grads += 1

            # send updated parameters
            self.put_weights(amount=consumed_grads)

    def learn(self, episodes: int, timesteps: int, *args, **kwargs):
        # create networks
        self._init_networks()

        # create workers
        self._init_workers(episodes, timesteps, *args, **kwargs)

        # put weights on queue
        self.put_weights(amount=self.num_actors)

        # start summary process
        self.on_start()

        # start workers
        for actor in self.workers:
            actor.start()

        # wait for gradients & send updated parameters
        self.update_and_send()

        # join workers
        for actor in self.workers:
            actor.join()
            # actor.terminate()

    def put_weights(self, amount: int):
        weights = (self.actor.get_weights(), self.critic.get_weights())

        for _ in range(amount):
            self.params_queue.put(np.copy(weights))

    def load_weights(self):
        pass

    def save_weights(self):
        pass

    def summary(self, init_networks=True):
        if init_networks:
            self._init_networks()

        self.actor.summary()
        self.critic.summary()


class Worker(mp.Process):

    def __init__(self, idx: int, env, params_queue: mp.Queue, grads_queue: mp.Queue, summary_queue: mp.Queue,
                 learn_args=(), learn_kwargs=None, **kwargs):
        assert isinstance(learn_args, tuple) and len(learn_args) >= 2
        super().__init__()

        self.idx = idx
        self.agent_kwargs = dict(env=env, name='a3c-worker', use_summary=False, **kwargs)
        self.agent: A3C = None

        self.params_queue = params_queue
        self.grads_queue = grads_queue
        self.summary_queue = summary_queue

        # parameters for `learn` method
        self.args = learn_args
        self.kwargs = learn_kwargs or {}
        self.max_timesteps = self.args[1]  # see: `Agent.learn(...)` definition

    def update(self):
        batch = self.agent.memory.get_data()

        # get gradients
        actor_grads = self.agent.actor.train_step(batch)
        critic_grads = self.agent.critic.train_step(batch)

        # push gradients
        self.grads_queue.put((actor_grads, critic_grads))

        # reset memory
        self.agent.memory.clear()

    def _init_agent(self):
        self.agent = A3C(**self.agent_kwargs)
        self.agent._init_networks()

        if self.agent.seed is not None:
            self.agent.set_random_seed(seed=self.idx * self.agent.seed)

        # hack: make `self` (Worker) behave like `agent` (A3C)
        self.agent.update = self.update
        self.agent.on_start = self.on_start

        self.agent.log_ = self.agent.log  # backup
        self.agent.log = self.log

        self.agent.on_transition_ = self.agent.on_transition  # backup
        self.agent.on_transition = self.on_transition

        if self.summary_queue is not None:
            # enable "summary" in workers if it's enabled in the main A3C agent
            self.agent.should_log = True
            self.agent.summary_queue = self.summary_queue

    def run(self):
        self._init_agent()
        self.agent.super.learn(*self.args, **self.kwargs)

    def sync_parameters(self):
        # wait for updated weights
        while self.params_queue.empty():
            time.sleep(self.agent.wait_time)

        # get and apply new weights
        weights = self.params_queue.get()

        self.agent.actor.set_weights(weights=weights[0])
        self.agent.critic.set_weights(weights=weights[1])

    def log(self, average=False, **kwargs):
        if self.agent.should_log:
            self.agent.log_(average=average, **({f'{k}:{self.idx}': v for k, v in kwargs.items()}))

    def on_start(self):
        # DON'T call super: this avoids an error about `self.agent.statistics.is_alive()`
        self.sync_parameters()

    def on_transition(self, transition, timestep: int, episode: int):
        self.agent.on_transition_(transition, timestep, episode)

        if transition['terminal'] or (timestep % self.agent.n_steps == 0) or (timestep == self.max_timesteps):
            terminal_state = self.agent.preprocess(transition['next_state'])

            if transition['terminal']:
                value = tf.zeros(shape=(1, 2), dtype=tf.float32)
            else:
                value = self.agent.critic(terminal_state, training=False)

            debug = self.agent.memory.end_trajectory(last_value=value)
            self.log(average=True, **debug)

            self.update()
            self.sync_parameters()


class GAEMemory(EpisodicMemory):

    def __init__(self, *args, agent: A3C, **kwargs):
        super().__init__(*args, **kwargs)

        if 'return' in self.data:
            raise ValueError('Key "return" is reserved!')

        if 'advantage' in self.data:
            raise ValueError('Key "advantage" is reserved!')

        self.data['return'] = np.zeros_like(self.data['value'])
        self.data['advantage'] = np.zeros(shape=(self.size, 1), dtype=np.float32)
        self.agent = agent

    def end_trajectory(self, last_value: tf.Tensor):
        value = last_value[:, 0] * tf.pow(10.0, last_value[:, 1])
        value = tf.expand_dims(value, axis=-1)

        rewards = tf.concat([self.data['reward'][:self.index], value], axis=0)
        values = tf.concat([self.data['value'][:self.index], last_value], axis=0)

        # value = base * 10^exponent
        v_base, v_exp = values[:, 0], values[:, 1]
        values = v_base * tf.pow(10.0, v_exp)
        values = tf.expand_dims(values, axis=-1)

        # compute returns and advantages for current episode
        returns = self.compute_returns(rewards)
        adv, advantages = self.compute_advantages(rewards, values)

        # store them
        self.data['return'][:self.index] = returns
        self.data['advantage'][:self.index] = advantages

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


@Network.register(name='A3C-ActorNetwork')
class ActorNetwork(PolicyNetwork):

    def train_step(self, batch: dict):
        """Performs a training step without applying the computed gradients"""
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

        if self.should_clip_gradients:
            gradients = self.clip_gradients(gradients, debug)

        return debug, gradients

    @tf.function
    def update(self, gradients):
        """Applies the given gradients"""
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


@Network.register(name='A3C-CriticNetwork')
class CriticNetwork(DecomposedValueNetwork):

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

        if self.should_clip_gradients:
            gradients = self.clip_gradients(gradients, debug)

        return debug, gradients

    @tf.function
    def update(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


if __name__ == '__main__':
    a3c = A3C(env='CartPole-v0', n_steps=5, parallel_actors=3, use_summary=True, seed=42)
    a3c.learn(1000 // 4, 200)
