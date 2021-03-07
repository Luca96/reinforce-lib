import os
import gym
import time
import json
import random
import multiprocessing as mp
import tensorflow as tf
import numpy as np

from rl import utils
from rl.parameters import DynamicParameter
from rl.environments import ParallelEnv

from rl.v2.memories import Memory, TransitionSpec

from typing import List, Dict, Union, Tuple


# TODO: provide a "fake" agent just for debugging components?
# TODO: gym.env wrapper that supports "reproucible" sampling, and has suitable "specs" for state and action-spaces
#  + basic pre-processing (e.g. conversion to numpy/tensor)?
# TODO: evaluation callbacks?
class Agent:
    """Agent abstract class"""
    # TODO: load, ...
    def __init__(self, env: Union[gym.Env, str], batch_size: int, gamma=0.99, seed=None, weights_dir='weights',
                 use_summary=True, drop_batch_remainder=True, skip_data=0, consider_obs_every=1, shuffle=True,
                 evaluation_dir='evaluation', shuffle_batches=False, traces_dir: str = None, repeat_action=1,
                 summary_keys: List[str] = None, name='agent'):
        assert batch_size >= 1
        assert repeat_action >= 1

        if isinstance(env, str):
            self.env = gym.make(env)
        else:
            self.env = env

        self.seed = None
        self.set_random_seed(seed)

        self.state_spec = utils.space_to_flat_spec(space=self.env.observation_space, name='state')
        self.action_spec = utils.space_to_flat_spec(space=self.env.action_space, name='action')

        self.gamma = tf.constant(gamma, dtype=tf.float32)
        self._memory = None
        self._dynamic_parameters = None
        self.repeat_action = int(repeat_action)

        self._init_action_space()

        # Record:
        if isinstance(traces_dir, str):
            self.should_record = True
            self.traces_dir = utils.makedir(traces_dir, name)
        else:
            self.should_record = False

        # Data option
        self.batch_size = int(batch_size)
        self.drop_batch_remainder = drop_batch_remainder
        self.skip_count = int(skip_data)
        self.obs_skipping = consider_obs_every
        self.shuffle_batches = shuffle_batches
        self.shuffle = shuffle

        self.data_args = dict(batch_size=self.batch_size, drop_remainder=drop_batch_remainder, skip=int(skip_data),
                              num_shards=int(consider_obs_every), shuffle_batches=shuffle_batches, shuffle=shuffle,
                              seed=self.seed)

        # Saving stuff (weights, config, evaluation):
        self.base_path = os.path.join(weights_dir, name)
        self.evaluation_path = utils.makedir(os.path.join(evaluation_dir, name))

        # JSON configuration file (keeps track of useful quantities, like dynamic-parameters' value)
        self.config_path = os.path.join(self.base_path, 'config.json')
        self.config = dict()

        # Statistics (tf.summary):
        if use_summary:
            self.summary_queue = mp.Queue()
            self.summary_stop_event = mp.Event()
            self.statistics = utils.SummaryProcess(self.summary_queue, self.summary_stop_event, name=name,
                                                   keys=summary_keys)
            self.should_log = True
        else:
            self.should_log = False

    @property
    def transition_spec(self) -> TransitionSpec:
        """Specifies what parts of a transition to store into Memory"""
        raise NotImplementedError

    @property
    def memory(self) -> Memory:
        """Defines the agent's internal memory"""
        raise NotImplementedError

    @property
    def dynamic_parameters(self) -> Dict[str, DynamicParameter]:
        if self._dynamic_parameters is None:
            self._dynamic_parameters = {}

            for name, value in self.__dict__.items():
                if isinstance(value, DynamicParameter):
                    self._dynamic_parameters[name] = value

        return self._dynamic_parameters

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

    def set_random_seed(self, seed):
        """Sets the random seed for tensorflow, numpy, python's random, and the environment"""
        if seed is not None:
            assert 0 <= seed < 2 ** 32

            tf.random.set_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            self.env.seed(seed)
            self.seed = seed
            print(f'Random seed {seed} set.')

    def act(self, state) -> Tuple[tf.Tensor, dict, dict]:
        """Agent prediction.
            - Returns a tuple: (action, other: dict, debug: dict)
            - `other` can contain any useful stuff (e.g.) about action, like its 'value' or 'log_prob'...
        """
        raise NotImplementedError

    def act_randomly(self, state) -> Tuple[tf.Tensor, dict, dict]:
        """Agent draws a random (or explorative) action.
            - Used in `explore()`
        """
        raise NotImplementedError

    def record(self, *args, **kwargs):
        pass

    def update(self):
        raise NotImplementedError

    def explore(self, steps: int):
        """Random exploration steps before training"""
        if steps <= 0:
            return

        state = self.env.reset()
        state = self.preprocess(state, evaluation=False)

        while steps > 0:
            action, other, debug = self.act_randomly(state)
            action_env = self.convert_action(action)

            for _ in range(self.repeat_action):
                next_state, reward, terminal, info = self.env.step(action=action_env)

                if terminal:
                    break

            transition = dict(state=state, action=action, reward=reward, next_state=next_state,
                              terminal=terminal, **(info or {}), **(other or {}))

            self.memory.store(transition)
            self.log(random_action_env=action_env, random_action=action, **debug)

            if terminal:
                state = self.env.reset()
                state = self.preprocess(state, evaluation=False)
                print(f'Explorative episode terminated. Steps left: {steps - 1}.')

            steps -= 1

    def learn(self, episodes: int, timesteps: int, render: Union[bool, int, None] = False, should_close=True,
              evaluation: Union[dict, bool] = None, exploration_steps=0, save=True):
        """Training loop"""
        assert episodes > 0
        assert timesteps > 0

        self.on_start()

        # Render:
        if render is True:
            render_freq = 1  # render each episode

        elif render is False or render is None:
            render_freq = episodes + 1  # never render
        else:
            render_freq = int(render)  # render at specified frequency

        # Evaluation:
        if isinstance(evaluation, dict):
            eval_freq = evaluation.pop('freq', episodes + 1)  # default: never evaluate
            assert isinstance(eval_freq, int)

            evaluation['should_close'] = False
            evaluation.setdefault('episodes', 1)  # default: evaluate on just 1 episode
            evaluation.setdefault('timesteps', timesteps)  # default: evaluate on the same number of timesteps
            evaluation.setdefault('render', render)  # default: same rendering options
        else:
            eval_freq = episodes + 1  # never evaluate

        # Saving:
        if save:
            # track 'average return' to determine best agent, also prefer newer agents (if equally good)
            best_return = -2**32
            should_save = True
        else:
            should_save = False

        # Exploration:
        self.explore(steps=int(exploration_steps))

        # Learning-loop:
        for episode in range(1, episodes + 1):
            self.on_episode_start(episode)

            should_render = episode % render_freq == 0
            should_evaluate = episode % eval_freq == 0

            episode_reward = 0.0
            t0 = time.time()

            state = self.env.reset()
            state = self.preprocess(state)

            for t in range(1, timesteps + 1):
                if should_render:
                    self.env.render()

                # Agent prediction
                action, other, debug = self.act(state)
                action_env = self.convert_action(action)

                # Environment step
                for _ in range(self.repeat_action):
                    next_state, reward, terminal, info = self.env.step(action=action_env)
                    episode_reward += reward

                    if terminal:
                        break

                # TODO: next_state not pre-processes, pre-process or avoid storing them!
                # TODO: provide `preprocess_transition()` fn?
                # transition = dict(state=state, action=action, reward=reward, next_state=next_state,
                #                   terminal=terminal, info=info, other=other)
                transition = dict(state=state, action=action, reward=reward, next_state=next_state,
                                  terminal=terminal, **(info or {}), **(other or {}))

                self.on_transition(transition, timestep=t, episode=episode)
                self.log(action_env=action_env, **debug)

                if terminal or (t == timesteps):
                    # TODO: put message in `on_episode_end`
                    print(f'Episode {episode} terminated after {t} timesteps in {round((time.time() - t0), 3)}s ' +
                          f'with reward {round(episode_reward, 3)}.')

                    self.log(timestep=t)
                    self.on_termination(last_transition=transition, timestep=t, episode=episode)
                    break
                else:
                    state = next_state
                    state = self.preprocess(state)

            self.on_episode_end(episode, episode_reward)

            if should_evaluate:
                eval_rewards = self.evaluate(**evaluation)

                self.log(eval_rewards=eval_rewards)
                print(f'[Evaluation] average return: {np.round(np.mean(eval_rewards), 2)}, '
                      f'std: {np.round(np.std(eval_rewards), 2)}')

                if should_save:
                    average_return = np.floor(np.mean(eval_rewards))

                    # TODO: when saving also save 'best_return' into the agent's config
                    if average_return >= best_return:
                        self.save()
                        best_return = average_return
                        print(f'Saved [{round(best_return, 2)}]')

            if self.should_record:
                self.record(episode)

        self.on_close(should_close)

    def evaluate(self, episodes: int, timesteps: int, render: Union[bool, int] = True, should_close=False) -> list:
        if render is True:
            render_freq = 1  # render each episode

        elif render is False or render is None:
            render_freq = episodes + 1  # never render
        else:
            render_freq = int(render)  # render at specified frequency

        episodic_rewards = [0.0] * episodes

        for episode in range(1, episodes + 1):
            self.on_episode_start(episode, evaluation=True)

            should_render = episode % render_freq == 0

            state = self.env.reset()
            state = self.preprocess(state, evaluation=True)

            for t in range(1, timesteps + 1):
                if should_render:
                    self.env.render()

                # Agent prediction
                action, other, debug = self.act(state)
                action_env = self.convert_action(action)

                # Environment step
                for _ in range(self.repeat_action):
                    next_state, reward, terminal, info = self.env.step(action=action_env)
                    episodic_rewards[episode - 1] += reward

                    if terminal:
                        break

                # TODO: log on evaluation?

                if terminal or (t == timesteps):
                    print(f'Evaluation episode {episode}/{episodes} terminated after {t}/{timesteps} timesteps ' +
                          f'with reward {round(episodic_rewards[episode - 1], 3)}.')
                    break
                else:
                    state = next_state
                    state = self.preprocess(state, evaluation=True)

            self.on_episode_end(episode, episode_reward=episodic_rewards[episode - 1], evaluation=True)

        if should_close:
            self.env.close()

        return episodic_rewards

    @classmethod
    def test(cls, args: dict, network_summary=False, **kwargs):
        """Rapid testing"""
        agent = cls(**kwargs)

        if network_summary:
            agent.summary()
            breakpoint()

        agent.learn(**args)

    def preprocess(self, state, evaluation=False):
        if isinstance(state, dict):
            return {f'state_{k}': utils.to_tensor(v, expand_axis=0) for k, v in state.items()}

        return utils.to_tensor(state, expand_axis=0)

    def log(self, average=False, **kwargs):
        if self.should_log:
            self.summary_queue.put(dict(average=average, **kwargs))

    def log_transition(self, transition: dict):
        self.log(reward=transition['reward'], action=transition['action'])

    def log_dynamic_parameters(self):
        self.log(**({k: p.value for k, p in self.dynamic_parameters.items()}))

    def summary(self):
        """Networks summary"""
        raise NotImplementedError

    def update_config(self, **kwargs):
        """Stores the given variables in the configuration dict for later saving"""
        for k, v in kwargs.items():
            self.config[k] = v

    def load_config(self):
        with open(self.config_path, 'r') as file:
            self.config = json.load(file)
            print('config loaded.')
            print(self.config)

    def save_config(self):
        with open(self.config_path, 'w') as file:
            json.dump(self.config, fp=file)
            print('config saved.')

    def reset(self):
        pass

    def load(self):
        """Loads the past agent's state"""
        self.load_weights()
        self.load_config()

    def save(self):
        """Saves the agent's state"""
        self.save_weights()
        self.save_config()

    def load_weights(self):
        raise NotImplementedError

    def save_weights(self):
        raise NotImplementedError

    def on_episode_start(self, episode: int, evaluation=False):
        """Called at the beginning of each episode"""
        self.reset()

    def on_episode_end(self, episode: int, episode_reward: float, evaluation=False):
        """Called *after* the end of each episode.
            - Used for memory-specific stuff.
            - Or, to update the agent (e.g. PPO)
        """
        for parameter in self.dynamic_parameters.values():
            parameter.on_episode()

        self.log(episode_reward=episode_reward)
        self.log_dynamic_parameters()

    # TODO: start/end ?
    def on_timestep(self, timestep: int):
        pass

    def on_transition(self, transition: dict, timestep: int, episode: int):
        """Called upon a transition occurs (agent prediction + environment step).
            - Useful for logging or pre-processing the transition.
            - Also for storing transition in memory, also conditional on actual timestep/episode num.
            - Or, updating the agent (e.g. DQN)
        """
        self.log_transition(transition)
        self.memory.store(transition)

    # TODO: delete
    def on_log(self, *args, **kwargs):
        """Called *after* environment step, usually used to log actions"""
        pass

    def on_termination(self, last_transition, timestep: int, episode: int):
        """Called *exactly* at the end of each episode, due to terminal state or maximum number of timesteps reached.
            - Used for agent-specific memory stuff.
            - Or, to update the agent (e.g. PPO)
        """
        pass

    def on_start(self):
        """Called *before* the training loop commence"""
        if self.should_log and not self.statistics.is_alive():
            self.statistics.start()

    def on_close(self, should_close: bool):
        """Called *after* the training loop ends"""
        self.close_summary()

        if should_close:
            print('closing...')
            self.env.close()

    def close_summary(self, timeout=2.0):
        """Closes the SummaryProcess"""
        if self.should_log:
            self.summary_stop_event.set()
            self.statistics.close(timeout)


# TODO: better name SynchronousAgent or SyncAgent?
class ParallelAgent(Agent):
    """Base class for Agents that uses parallel environments (e.g. A2C, PPO, ...)"""
    def __init__(self, env, *args, num_actors: int, name='parallel-agent', **kwargs):
        assert num_actors >= 1

        self.num_actors = int(num_actors)
        self.max_timesteps = 0  # being init in `self.learn(...)`

        super().__init__(env=ParallelEnv(env, num=self.num_actors), *args, name=name, **kwargs)

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

    # TODO: `evaluation` and `exploration`
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

            for state in states:
                for k, v in state.items():
                    states_[f'state_{k}'].append(v)

            return states_

        return states

    def log_transition(self, transition: Dict[str, list]):
        data = dict()

        for i, (reward, action) in enumerate(zip(transition['reward'], transition['action'])):
            data[f'reward_{i}'] = reward
            data[f'action_{i}'] = action

        self.log(**data)


class AsyncAgent(Agent):
    pass


class RandomAgent(Agent):

    def __init__(self, *args, name='random-agent', **kwargs):
        super().__init__(*args, name=name, **kwargs)

        action_space = self.env.action_space

        if isinstance(action_space, gym.spaces.Discrete):
            # sample action from categorical distribution with uniform probabilities
            logits = tf.ones(shape=(1, action_space.n), dtype=tf.int32)
            self.sample = lambda _: tf.random.categorical(logits=logits, num_samples=1, seed=self.seed)

        elif isinstance(action_space, gym.spaces.Box) and action_space.is_bounded():
            self.sample = lambda _: tf.random.uniform(shape=action_space.shape, minval=action_space.low,
                                                      maxval=action_space.high, seed=self.seed)
        else:
            raise ValueError('Only bounded environments are supported: Discrete, and Box.')

        self.convert_action = lambda a: tf.squeeze(a).numpy()

    def act(self, state) -> Tuple[tf.Tensor, dict, dict]:
        return self.sample(state), {}, {}

    def learn(self, *args, **kwargs):
        raise RuntimeError('RandomAgent does not support learning.')
