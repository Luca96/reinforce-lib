"""Advantage Actor-Critic (A2C)"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import numpy as np

from rl import utils
from rl.parameters import DynamicParameter

from rl.v2.agents.agents import ParallelAgent
from rl.v2.agents.ppo import GlobalMemory
from rl.v2.memories import TransitionSpec
from rl.v2.networks import Network, ValueNetwork
from rl.v2.networks.policies import PolicyNetwork

from typing import Tuple, Union


class ActorNetwork(PolicyNetwork):

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
    #     return tf.identity(distribution), distribution.mean(), distribution.stddev()

    def structure(self, inputs, name='A2C-ActorNetwork', **kwargs) -> tuple:
        return super().structure(inputs, name=name, **kwargs)


class A2C(ParallelAgent):
    """A2C agent"""

    def __init__(self, env, horizon: int, gamma=0.99, name='a2c-agent', optimizer='adam',
                 actor_lr: utils.DynamicType = 1e-3, critic_lr: utils.DynamicType = 3e-4, clip_norm=(None, None),
                 lambda_=1.0, num_actors=16, entropy: utils.DynamicType = 0.01, actor: dict = None, critic: dict = None,
                 **kwargs):
        assert horizon >= 1

        super().__init__(env, num_actors=num_actors, batch_size=horizon * num_actors, gamma=gamma,
                         name=name, **kwargs)

        # Hyper-parameters:
        self.horizon = int(horizon)
        self.lambda_ = tf.constant(lambda_, dtype=tf.float32)
        self.entropy_strength = DynamicParameter.create(value=entropy)

        self.adv_scale = DynamicParameter.create(value=1.0)
        self.adv_normalization_fn = utils.get_normalization_fn(arg='identity')
        self.returns_norm_fn = utils.get_normalization_fn(arg='identity')

        self.actor_lr = DynamicParameter.create(value=actor_lr)
        self.critic_lr = DynamicParameter.create(value=critic_lr)

        # Networks
        self.weights_path = dict(policy=os.path.join(self.base_path, 'actor'),
                                 value=os.path.join(self.base_path, 'critic'))

        self.actor = Network.create(agent=self, **(actor or {}), base_class=ActorNetwork)
        self.critic = Network.create(agent=self, **(critic or {}), base_class=ValueNetwork)

        if clip_norm is None:
            clip_norm = (None, None)

        self.actor.compile(optimizer, clip_norm=clip_norm[0], clip=self.clip_grads, learning_rate=self.actor_lr)
        self.critic.compile(optimizer, clip_norm=clip_norm[1], clip=self.clip_grads, learning_rate=self.critic_lr)

    @property
    def transition_spec(self) -> TransitionSpec:
        return TransitionSpec(state=self.state_spec, action=(self.num_actions,), next_state=False, terminal=False,
                              reward=(1,), other=dict(value=(1,)))

    def define_memory(self) -> GlobalMemory:
        return GlobalMemory(self.transition_spec, agent=self, shape=self.horizon * self.num_actors)

    @tf.function
    def act(self, states, **kwargs) -> Tuple[tf.Tensor, dict, dict]:
        actions, mean, std = self.actor(states, training=False)
        values = self.critic(states, training=False)

        other = dict(value=values)
        debug = dict(distribution_mean=tf.reduce_mean(mean), distribution_std=tf.reduce_mean(std))

        return actions, other, debug

    @tf.function
    def act_evaluation(self, state, **kwargs):
        actions, _, _ = self.actor(state, training=False, deterministic=True, **kwargs)
        return actions

    # def learn(self, episodes: int, timesteps: int, render: Union[bool, int, None] = False, should_close=True,
    #           evaluation: Union[dict, bool] = None, exploration_steps=0, save=True):
    #     assert episodes > 0
    #     assert timesteps > 0
    #
    #     import time
    #     t0 = time.time()
    #     total_sec = 0
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
    #     for episode in range(1, episodes + 1):
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
    #                 total_sec += seconds
    #
    #                 print(f'Episode {episode} terminated after {t} timesteps in {round(seconds, 3)}s ' +
    #                       f'with reward {round(episode_reward, 3)}')
    #
    #                 self.log(timestep=t, total_steps=self.total_steps, time_seconds=total_sec,
    #                          seconds_per_epoch=seconds)
    #                 self.on_termination(last_transition=transition)
    #                 break
    #             else:
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
                    terminal_states[i] = info.pop('__terminal_state')

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

    # def load_weights(self):
    #     self.actor.load_weights(filepath=self.weights_path['actor'], by_name=False)
    #     self.critic.load_weights(filepath=self.weights_path['critic'], by_name=False)
    #
    # def save_weights(self, path: str):
    #     self.actor.save_weights(filepath=os.path.join(path, 'actor'))
    #     self.critic.save_weights(filepath=os.path.join(path, 'critic'))
    #
    # def summary(self):
    #     self.actor.summary()
    #     self.critic.summary()


if __name__ == '__main__':
    import gym
    utils.set_random_seed(42)

    # # achieves 390+ reward
    # agent = A2C(env='CartPole-v1', name='a2c-cart_v1', horizon=16, num_actors=16,
    #             actor_lr=3e-4, entropy=1e-4, clip_norm=None, use_summary=True,
    #             actor=dict(units=32), critic=dict(units=64),
    #             seed=utils.GLOBAL_SEED, parallel_env=SequentialEnv)
    #
    # agent.learn(episodes=250, timesteps=500, save=True, render=False, should_close=True,
    #             evaluation=dict(episodes=25, freq=10))
    # exit()

    # agent = A2C(env='Pendulum-v1', name='a2c-pendulum', horizon=64, num_actors=8,
    #             actor_lr=1e-3, critic_lr=1e-3, entropy=1e-3, clip_norm=None, use_summary=True,
    #             actor=dict(units=128), critic=dict(units=128),
    #             seed=utils.GLOBAL_SEED, parallel_env=SequentialEnv)

    # agent.learn(episodes=250, timesteps=500, save=False, render=False, should_close=True,
    #             evaluation=dict(episodes=20, freq=5))

    from rl.parameters import LinearDecay, ExponentialDecay

    agent = A2C(env='LunarLanderContinuous-v2', name='a2c-lunar_c', horizon=16*2, num_actors=16,
                actor_lr=3e-4, critic_lr=1e-3, clip_norm=(2.5, 10.0), use_summary=True and False,
                entropy=LinearDecay(1.0, end_value=0.0, steps=400),
                # entropy=ExponentialDecay(1.0, steps=500, rate=0.99),
                actor=dict(units=128, num_layers=4), critic=dict(units=128), lambda_=0.95,
                seed=utils.GLOBAL_SEED)

    def convert_action():
        space = agent.env.action_space
        assert isinstance(space, gym.spaces.Box) and space.is_bounded()

        low = tf.constant(space.low, dtype=tf.float32)
        delta = tf.constant(space.high - space.low, dtype=tf.float32)

        return lambda actions: [np.squeeze(a * delta + low) for a in actions]  # lunar-c
        # return lambda actions: [np.reshape(a * delta + low, newshape=1) for a in actions]  # pendulum

    agent.convert_action = convert_action()

    # agent.summary()
    # breakpoint()

    # agent.load()

    # runs = [agent.record(timesteps=1000, force=True, rename=True) for _ in range(50)]  # [(reward, path)]
    # runs = sorted(runs, key=lambda x: x[0])  # sort by reward
    # runs = runs[:-10]  # keep only best 10, so remove the others
    #
    # for _, path in runs:
    #     utils.remove_folder(path)
    #
    # exit()

    agent.learn(episodes=500, timesteps=1000, save=True, render=False, should_close=True,
                evaluation=dict(episodes=25, render=20, freq=10))
