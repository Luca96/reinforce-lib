"""Synchronous Advantage Actor-Critic (A2C)"""

import os
import gym
import time
import tensorflow as tf
import numpy as np

from rl import utils
from rl.parameters import DynamicParameter

from rl.v2.agents import Agent
from rl.v2.memories import TransitionSpec
from rl.v2.memories.episodic import EpisodicMemory
from rl.v2.networks import ValueNetwork, PolicyNetwork

from rl.environments.gym.parallel import ParallelEnv

from typing import List, Tuple, Union, Dict


# TODO: implement `evaluation` and `exploration`
# TODO: check update method
# TODO: `tf.reduce_sum` instead of `tf.reduce_mean` in both actor's and critic's objective?
class A2C(Agent):
    """Sequential (single-process) implementation of A2C"""

    def __init__(self, env, name='a2c-agent', parallel_actors=16, n_steps=5, entropy=0.01, load=False,
                 optimizer='rmsprop', lambda_=0.95, normalize_advantages: Union[None, str] = 'sign', actor: dict = None,
                 critic: dict = None, advantage_scale: utils.DynamicType = 1.0, actor_lr: utils.DynamicType = 7e-4,
                 clip_norm: Tuple[utils.DynamicType] = (1.0, 1.0), critic_lr: utils.DynamicType = 7e-4, **kwargs):
        assert n_steps >= 1
        assert parallel_actors >= 1

        self.n_actors = int(parallel_actors)
        self.n_steps = int(n_steps)

        super().__init__(env=ParallelEnv(env, num=self.n_actors), batch_size=self.n_steps, name=name, **kwargs)

        self._init_action_space()
        self.max_timesteps = 0  # being init in `self.learn(...)`

        self.lambda_ = tf.constant(lambda_, dtype=tf.float32)
        self.entropy_strength = DynamicParameter.create(value=entropy)
        self.adv_scale = DynamicParameter.create(value=advantage_scale)
        self.adv_normalization_fn = utils.get_normalization_fn(name=normalize_advantages)

        self.actor_lr = DynamicParameter.create(value=actor_lr)
        self.critic_lr = DynamicParameter.create(value=critic_lr)

        # shared networks (and optimizer)
        self.actor = ActorNetwork(agent=self, log_prefix='actor', **(actor or {}))
        self.critic = CriticNetwork(agent=self, log_prefix='critic', **(critic or {}))

        if isinstance(optimizer, dict):
            opt_args = optimizer
            optimizer = opt_args.pop('name', 'rmsprop')
        else:
            opt_args = {}

        self.actor.compile(optimizer, clip_norm=clip_norm[0], learning_rate=self.actor_lr, **opt_args)
        self.critic.compile(optimizer, clip_norm=clip_norm[1], learning_rate=self.critic_lr, **opt_args)

        self.weights_path = dict(policy=os.path.join(self.base_path, 'actor'),
                                 value=os.path.join(self.base_path, 'critic'))

        if load:
            self.load()

    @property
    def transition_spec(self) -> TransitionSpec:
        state_spec = {k: (self.n_steps,) + shape for k, shape in self.state_spec.items()}

        return TransitionSpec(state=state_spec, action=(self.n_steps, self.num_actions), next_state=False,
                              terminal=False, reward=(self.n_steps, 1), other=dict(value=(self.n_steps, 1)))

    @property
    def memory(self) -> 'ParallelGAEMemory':
        if self._memory is None:
            self._memory = ParallelGAEMemory(self.transition_spec, agent=self, size=self.n_actors)

        return self._memory

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

                def convert_action(actions) -> list:
                    return [tf.squeeze(a * self.action_range + self.action_low).numpy() for a in actions]

                self.convert_action = convert_action
            else:
                self.distribution_type = 'gaussian'
                self.convert_action = lambda actions: [tf.squeeze(a).numpy() for a in actions]
        else:
            # discrete:
            assert isinstance(action_space, gym.spaces.Discrete)
            self.distribution_type = 'categorical'

            self.num_actions = 1
            self.num_classes = action_space.n
            self.convert_action = lambda actions: [tf.cast(tf.squeeze(a), dtype=tf.int32).numpy() for a in actions]

    def act(self, states) -> Tuple[tf.Tensor, dict, dict]:
        actions, _, means, std = self.actor(states, training=False)
        values = self.critic(states, training=False)

        other = dict(value=values)
        debug = dict()

        if self.distribution_type != 'categorical':
            for i, (mu, sigma) in enumerate(zip(means, std)):
                debug[f'distribution_mean_{i}'] = mu
                debug[f'distribution_std_{i}'] = sigma

        return actions, other, debug

    @staticmethod
    def average_gradients(gradients_list) -> list:
        n = 1.0 / len(gradients_list)

        gradients = gradients_list[0]

        for i in range(1, len(gradients_list)):
            grads = gradients_list[i]

            for j, g in enumerate(grads):
                gradients[j] += g

        return [g * n for g in gradients]

    # def update(self):
    #     batches = self.memory.get_data()
    #
    #     # get gradients for each batch (actor's data)
    #     actor_grads = [self.actor.train_step(batch) for batch in batches]
    #     critic_grads = [self.critic.train_step(batch) for batch in batches]
    #
    #     # update weights on averaged gradients across actors
    #     self.actor.update(gradients=self.average_gradients(actor_grads))
    #     self.critic.update(gradients=self.average_gradients(critic_grads))

    def update(self):
        batches = self.memory.get_data()

        all_batches = {k: v for k, v in batches[0].items()}
        for i in range(1, len(batches)):
            for k, v in batches[i].items():
                all_batches[k] = tf.concat([v, all_batches[k]], axis=0)

        actor_grads = self.actor.train_step(all_batches)
        critic_grads = self.critic.train_step(all_batches)

        # update weights
        self.actor.update(gradients=actor_grads)
        self.critic.update(gradients=critic_grads)

    # TODO: evaluation
    def learn(self, episodes: int, timesteps: int, render: Union[bool, int, None] = False, should_close=True,
              evaluation: Union[dict, bool] = None, exploration_steps=0, save=True):
        """Training loop"""
        assert episodes > 0
        assert timesteps > 0

        tk = time.time()
        tot_rew = 0.0

        # Render:
        if render is True:
            render_freq = 1  # render each episode

        elif render is False or render is None:
            render_freq = episodes + 1  # never render
        else:
            render_freq = int(render)  # render at specified frequency

        # # Evaluation:
        # if isinstance(evaluation, dict):
        #     eval_freq = evaluation.pop('freq', episodes + 1)  # default: never evaluate
        #     assert isinstance(eval_freq, int)
        #
        #     evaluation['should_close'] = False
        #     evaluation.setdefault('episodes', 1)  # default: evaluate on just 1 episode
        #     evaluation.setdefault('timesteps', timesteps)  # default: evaluate on the same number of timesteps
        #     evaluation.setdefault('render', render)  # default: same rendering options
        # else:
        #     eval_freq = episodes + 1  # never evaluate

        # # Saving:
        # if save:
        #     # track 'average return' to determine best agent, also prefer newer agents (if equally good)
        #     best_return = -2 ** 32
        #     should_save = True
        # else:
        #     should_save = False

        # # Exploration:
        # self.explore(steps=int(exploration_steps))

        self.max_timesteps = timesteps
        self.on_start()

        # Learning-loop:
        for episode in range(1, episodes + 1):
            self.on_episode_start(episode)

            should_render = episode % render_freq == 0
            # should_evaluate = episode % eval_freq == 0

            episode_reward = 0.0
            t0 = time.time()

            states = self.env.reset()
            states = self.preprocess(states)

            for t in range(1, timesteps + 1):
                if should_render:
                    self.env.render()

                # Agent prediction
                actions, other, debug = self.act(states)
                actions_env = self.convert_action(actions)

                # Environment step
                for _ in range(self.repeat_action):
                    next_states, rewards, terminals, info = self.env.step(actions=actions_env)
                    episode_reward += np.mean(rewards)

                    if any(terminals):
                        break

                transition = dict(state=states, action=actions, reward=rewards, next_state=next_states,
                                  terminal=terminals, **(info or {}), **(other or {}))

                self.on_transition(transition, timestep=t, episode=episode)
                self.log(action_env=actions_env, **debug)

                if any(terminals) or (t == timesteps):
                    print(f'Episode {episode} terminated after {t} timesteps in {round((time.time() - t0), 3)}s ' +
                          f'with reward {round(episode_reward, 3)}.')
                    tot_rew += episode_reward

                    self.log(timestep=t)
                    self.on_termination(last_transition=transition, timestep=t, episode=episode)
                    break
                else:
                    states = next_states
                    states = self.preprocess(states)

            self.on_episode_end(episode, episode_reward)

            # if should_evaluate:
            #     eval_rewards = self.evaluate(**evaluation)
            #
            #     self.log(eval_rewards=eval_rewards)
            #     print(f'[Evaluation] average return: {np.round(np.mean(eval_rewards), 2)}, '
            #           f'std: {np.round(np.std(eval_rewards), 2)}')
            #
            #     if should_save:
            #         average_return = np.floor(np.mean(eval_rewards))
            #
            #         # TODO: when saving also save 'best_return' into the agent's config
            #         if average_return >= best_return:
            #             self.save()
            #             best_return = average_return
            #             print(f'Saved [{round(best_return, 2)}]')

            if self.should_record:
                self.record(episode)

        print(f'Time taken {round(time.time() - tk, 3)}s.')
        print(f'Total episodic reward: {round(tot_rew, 3)}')
        self.on_close(should_close)

    def preprocess(self, states, evaluation=False):
        if isinstance(states[0], dict):
            states_ = {f'state_{k}': [v] for k, v in states[0].items()}

            for i in range(1, len(states)):
                for k, v in states[i].items():
                    states_[f'state_{k}'].append(v)

            return states_

        return states

    def on_transition(self, transition: Dict[str, list], timestep: int, episode: int):
        super().on_transition(transition, timestep, episode)

        if any(transition['terminal']) or (timestep % self.n_steps == 0) or (timestep == self.max_timesteps):
            terminal_states = self.preprocess(transition['next_state'])

            values = self.critic(terminal_states, training=False)
            values = values * utils.to_float(tf.logical_not(transition['terminal']))

            debug = self.memory.end_trajectory(last_values=values)
            self.log(average=True, **debug)

            self.update()
            self.memory.clear()

    def log_transition(self, transition: Dict[str, list]):
        data = dict()

        for i, (reward, action) in enumerate(zip(transition['reward'], transition['action'])):
            data[f'reward_{i}'] = reward
            data[f'action_{i}'] = action

        self.log(**data)

    def load_weights(self):
        self.actor.load_weights(filepath=self.weights_path['actor'], by_name=False)
        self.critic.load_weights(filepath=self.weights_path['critic'], by_name=False)

    def save_weights(self):
        self.actor.save_weights(filepath=self.weights_path['actor'])
        self.critic.save_weights(filepath=self.weights_path['critic'])

    def summary(self):
        self.actor.summary()
        self.critic.summary()


class ActorNetwork(PolicyNetwork):

    def train_step(self, batch: dict):
        if isinstance(batch, tuple):
            batch = batch[0]

        debug, grads = self.train_on_batch(batch)
        self.agent.log(average=True, **({f'{self.prefix}_{k}': v for k, v in debug.items()}))

        return grads

    @tf.function
    def train_on_batch(self, batch):
        with tf.GradientTape() as tape:
            loss, debug = self.objective(batch)

        gradients = tape.gradient(loss, self.trainable_variables)
        debug['gradient_norm'] = [tf.norm(g) for g in gradients]

        if self.should_clip_gradients:
            gradients, global_norm = utils.clip_gradients2(gradients, norm=self.clip_norm())
            debug['gradient_clipped_norm'] = [tf.norm(g) for g in gradients]
            debug['gradient_global_norm'] = global_norm
            debug['clip_norm'] = self.clip_norm.value

        return debug, gradients

    @tf.function
    def update(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class CriticNetwork(ValueNetwork):

    def train_step(self, batch: dict):
        if isinstance(batch, tuple):
            batch = batch[0]

        debug, grads = self.train_on_batch(batch)
        self.agent.log(average=True, **({f'{self.prefix}_{k}': v for k, v in debug.items()}))

        return grads

    @tf.function
    def train_on_batch(self, batch):
        with tf.GradientTape() as tape:
            loss, debug = self.objective(batch)

        gradients = tape.gradient(loss, self.trainable_variables)
        debug['gradient_norm'] = [tf.norm(g) for g in gradients]

        if self.should_clip_gradients:
            gradients, global_norm = utils.clip_gradients2(gradients, norm=self.clip_norm())
            debug['gradient_clipped_norm'] = [tf.norm(g) for g in gradients]
            debug['gradient_global_norm'] = global_norm
            debug['clip_norm'] = self.clip_norm.value

        return debug, gradients

    @tf.function
    def update(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class ParallelGAEMemory(EpisodicMemory):

    def __init__(self, *args, agent: A2C, **kwargs):
        super().__init__(*args, **kwargs)

        if 'return' in self.data:
            raise ValueError('Key "return" is reserved!')

        if 'advantage' in self.data:
            raise ValueError('Key "advantage" is reserved!')

        self.data['return'] = np.zeros_like(self.data['value'])
        self.data['advantage'] = np.zeros(shape=(self.size, agent.n_steps, 1), dtype=np.float32)
        self.agent = agent

    def _store(self, data, spec, key, value):
        if not isinstance(value, dict):
            array = np.asanyarray(value, dtype=np.float32)
            array = np.reshape(array, newshape=(self.size, spec['shape'][-1]))  # TODO: check spec['shape']

            # indexing: key, env, index (timestep)
            #   - `array` has shape (n_envs, n_steps)
            #   - each `v` in `array` is data for the corresponding env
            for env_index, v in enumerate(array):
                data[key][env_index][self.index] = v
        else:
            for k, v in value.items():
                self._store(data=data[key], spec=spec[k], key=k, value=v)

    def end_trajectory(self, last_values) -> dict:
        debug = dict()
        data_reward, data_value = self.data['reward'], self.data['value']
        data_return, data_adv = self.data['return'], self.data['advantage']

        for i in range(self.agent.n_actors):
            v = tf.expand_dims(last_values[i], axis=-1)
            rewards = np.concatenate([data_reward[i][:self.index], v], axis=0)
            values = np.concatenate([data_value[i][:self.index], v], axis=0)

            # compute returns and advantages for i-th environment
            returns = self.compute_returns(rewards)
            adv, advantages = self.compute_advantages(rewards, values)

            # store them
            data_return[i][:self.index] = returns
            data_adv[i][:self.index] = advantages

            # debug
            debug[f'returns_{i}'] = returns
            debug[f'advantages_normalized_{i}'] = advantages
            debug[f'advantages_{i}'] = adv
            debug[f'values_{i}'] = values

        return debug

    def compute_returns(self, rewards):
        returns = utils.rewards_to_go(rewards, discount=self.agent.gamma)
        return returns

    def compute_advantages(self, rewards, values):
        advantages = utils.gae(rewards, values=values, gamma=self.agent.gamma, lambda_=self.agent.lambda_)
        norm_adv = self.agent.adv_normalization_fn(advantages) * self.agent.adv_scale()
        return advantages, norm_adv

    def get_data(self) -> List[dict]:
        """Returns a batch of data for each environment/actor"""
        if self.full:
            index = self.size
        else:
            index = self.index

        n_envs = self.agent.n_actors

        def _get(data_list, _k, _v):
            if not isinstance(_v, dict):
                for i in range(n_envs):
                    data_list[i][_k] = _v[i][:index]
            else:
                for data in data_list:
                    data[_k] = dict()

                for k, v in _v.items():
                    _get([data[_k] for data in data_list], k, v)

        batches = [dict() for _ in range(n_envs)]

        for key, value in self.data.items():
            _get(batches, key, value)

        return batches


if __name__ == '__main__':
    a2c = A2C(env='CartPole-v0', n_steps=5, use_summary=True, seed=42)
    a2c.learn(1000//4, 200)
