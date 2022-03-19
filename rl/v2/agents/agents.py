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
from rl.environments.gym.parallel import AbstractParallelEnv, SequentialEnv

from rl.v2.memories import Memory, TransitionSpec

from typing import List, Dict, Union, Tuple


# TODO: keep in mind: https://github.com/openai/gym-recording
# TODO: env.seed will be deprecated, use env.reset(seed=seed) instead..
# TODO: compile anything numpy-related with `numba`:
#  https://towardsdatascience.com/supercharging-numpy-with-numba-77ed5b169240
# TODO: when saving agent weights, also save a "preset" file with the hyper-parameters so that the agent can be loaded.
# TODO: rename `repeat_action` to `frame-skip`?
# TODO: when frame-skip > 1, adjust the discount factor i.e. to \gamma^n
# TODO: summary of agent hyper-parameters?
# TODO: evaluation callbacks?
class Agent:
    """Agent abstract class"""
    # TODO: load, ...
    def __init__(self, env: Union[gym.Env, str], batch_size: int, gamma=0.99, seed=None, save_dir='weights',
                 use_summary=True, drop_batch_remainder=True, skip_data=0, consider_obs_every=1, shuffle=True,
                 evaluation_dir='evaluation', shuffle_batches=False, traces_dir: str = None, repeat_action=1,
                 summary: dict = None, name='agent', reward_scale=1.0, clip_grads='global', clip_rewards: tuple = None):
        assert batch_size >= 1
        assert repeat_action >= 1

        if isinstance(env, str):
            self.env = gym.make(env)
        else:
            self.env = env

        self.seed = None
        self.set_random_seed(seed or utils.GLOBAL_SEED)
        self.name = str(name)

        self.state_spec = utils.space_to_flat_spec(space=self.env.observation_space, name='state')
        self.action_spec = utils.space_to_flat_spec(space=self.env.action_space, name='action')

        # TODO: better support for train/eval/explore flags
        self.is_learning = False
        self.is_evaluating = False
        self.is_exploring = False

        # keep track of episode and timestep number during training
        self.episode = 0
        self.timestep = 0
        self.max_timesteps = None  # being init in `on_start()`
        self.total_steps = 0  # total number of interaction steps on the whole training

        self.gamma = tf.constant(gamma, dtype=tf.float32)
        self._memory = None
        self._networks = None
        self._dynamic_parameters = None
        self.clip_grads = clip_grads
        self.repeat_action = int(repeat_action)

        self._init_action_space()

        # Reward stuff:
        self.reward_scale = DynamicParameter.create(value=reward_scale)
        self._init_reward_clipping(clip_rewards)

        # Record:
        if isinstance(traces_dir, str):
            self.should_record = True
            self.traces_dir = utils.makedir(traces_dir, self.name)
        else:
            self.should_record = False

        # Data option
        self.batch_size = int(batch_size)
        self.drop_batch_remainder = drop_batch_remainder
        self.skip_count = int(skip_data)
        self.obs_skipping = consider_obs_every
        self.shuffle_batches = shuffle_batches
        self.shuffle = shuffle

        # TODO: deprecate?
        self.data_args = dict(batch_size=self.batch_size, drop_remainder=drop_batch_remainder, skip=int(skip_data),
                              num_shards=int(consider_obs_every), shuffle_batches=shuffle_batches, shuffle=shuffle,
                              seed=self.seed)

        # Saving stuff (weights, config, evaluation):
        self.save_dir = str(save_dir)
        self.base_path = os.path.join(self.save_dir, self.name, utils.actual_datetime())
        self.evaluation_path = utils.makedir(os.path.join(evaluation_dir, self.name))

        # JSON configuration file (keeps track of useful quantities, like dynamic-parameters' value)
        self.config_path = os.path.join(self.base_path, 'config.json')
        self.config = dict()

        # Statistics (tf.summary):
        if use_summary:
            self.summary_queue = mp.Queue()
            self.summary_stop_event = mp.Event()
            self.statistics = utils.SummaryProcess(self.summary_queue, self.summary_stop_event, name=name,
                                                   **(summary or {}))
            self.should_log = True
        else:
            self.should_log = False
    
    @classmethod
    def from_preset(cls, preset: dict, **kwargs) -> 'Agent':
        """Creates an Agent instance (type depends on caller class) from given `preset` (i.e. configuration)"""
        args = preset.copy()
        args.update(**kwargs)

        return cls(**args)

    @property
    def transition_spec(self) -> TransitionSpec:
        """Specifies what parts of a transition to store into Memory"""
        raise NotImplementedError

    @property
    def memory(self) -> Memory:
        """Defines the agent's internal memory"""
        if self._memory is None:
            self._memory = self.define_memory()

        return self._memory

    def define_memory(self) -> Memory:
        """Definition of the agent's internal memory buffer"""
        raise NotImplementedError

    @property
    def networks(self) -> Dict:
        """A dictionary storing all the network instances, used for weight loading and saving."""
        if self._networks is None:
            from rl.v2.networks import Network
            self._networks = {k: v for k, v in self.__dict__.items() if isinstance(v, Network)}

        return self._networks

    @property
    def dynamic_parameters(self) -> Dict[str, DynamicParameter]:
        if self._dynamic_parameters is None:
            self._dynamic_parameters = {k: v for k, v in self.__dict__.items() if isinstance(v, DynamicParameter)}

        return self._dynamic_parameters

    # TODO: edit
    def _init_action_space(self):
        action_space = self.env.action_space

        if isinstance(action_space, gym.spaces.Box):
            self.num_actions = action_space.shape[0]
            self.discrete_actions = False

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
            self.discrete_actions = True

            self.num_actions = 1
            self.num_classes = action_space.n
            self.convert_action = lambda a: tf.cast(tf.squeeze(a), dtype=tf.int32).numpy()

    def _init_reward_clipping(self, clip_rewards):
        if clip_rewards is None:
            self.reward_clip_range = (-np.inf, np.inf)

        elif isinstance(clip_rewards, (tuple, list)) and len(clip_rewards) >= 1:
            low = clip_rewards[0]

            if (low is None) or (low == -np.inf) or (low == np.nan):
                low = -np.inf

            elif isinstance(low, (int, float)):
                low = float(low)
            else:
                raise ValueError(f'Lower-bound for reward clipping should be [None, -np.inf, np.nan, int, float] not'
                                 f' {type(low)}.')

            if len(clip_rewards) >= 2:
                high = clip_rewards[1]

                if (high is None) or (high == np.inf) or (high == np.nan):
                    high = np.inf

                elif isinstance(high, (int, float)):
                    high = float(high)
                else:
                    raise ValueError(
                        f'Higher-bound for reward clipping should be [None, np.inf, np.nan, int, float] not'
                        f' {type(high)}.')
            else:
                high = np.inf

            self.reward_clip_range = (low, high)
        else:
            raise ValueError(f'Parameter "clip_rewards" should be None, list or tuple not {type(clip_rewards)}.')

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

    def set_hyper(self, *args, **kwargs):
        raise NotImplementedError

    def act(self, state, deterministic=False, inference=False, **kwargs) -> Tuple[tf.Tensor, dict, dict]:
        """Agent prediction.
            - Returns a tuple: (action, other: dict, debug: dict)
            - `other` can contain any useful stuff (e.g.) about action, like its 'value' or 'log_prob'...
            - if `deterministic=True`, agent will exploit its knowledge, predicting optimal actions
            - if `inference=True`, corresponding layers are enabled only at interaction time.
        """
        raise NotImplementedError

    def act_randomly(self, state) -> Tuple[tf.Tensor, dict, dict]:
        """Agent draws a random (or exploratory) action.
            - Used in `explore()`
        """
        raise NotImplementedError

    def record(self, timesteps: int, folder='video', rename=True, **kwargs) -> Tuple[float, str]:
        from gym.wrappers import RecordVideo

        timestamp = utils.actual_datetime()
        record_path = os.path.join(folder, self.name, timestamp)

        env = RecordVideo(env=self.env, video_folder=record_path, **kwargs)
        episode_reward = 0.0

        # start recording
        state = env.reset()
        state = self.preprocess(state, evaluation=True)

        t = 0
        while t < timesteps + 1:
            t += 1

            # Agent prediction
            action, _, _ = self.act(state, deterministic=True)
            action_env = self.convert_action(action)

            # Environment step
            for _ in range(self.repeat_action):
                next_state, reward, terminal, info = env.step(action=action_env)
                episode_reward += reward

                if terminal:
                    break

            if terminal or (t == timesteps):
                print(f'Record episode terminated after {t}/{timesteps} timesteps ' +
                      f'with reward {round(episode_reward, 3)}.')
                break
            else:
                state = next_state
                state = self.preprocess(state, evaluation=True)

        env.close_video_recorder()

        if rename:
            new_path = os.path.join(folder, self.name, f'{timestamp}-r{round(episode_reward, 2)}')

            utils.copy_folder(src=record_path, dst=new_path)
            utils.remove_folder(path=record_path)

            record_path = new_path

        return episode_reward, record_path

    def update(self):
        raise NotImplementedError

    # def explore(self, steps: int):
    #     """Random exploration steps before training"""
    #     if steps <= 0:
    #         return
    #
    #     state = self.env.reset()
    #     state = self.preprocess(state, evaluation=False)
    #
    #     while steps > 0:
    #         action, other, debug = self.act_randomly(state)
    #         action_env = self.convert_action(action)
    #
    #         for _ in range(self.repeat_action):
    #             next_state, reward, terminal, info = self.env.step(action=action_env)
    #
    #             if terminal:
    #                 break
    #
    #         transition = dict(state=state, action=action, reward=reward, next_state=next_state,
    #                           terminal=terminal, **(info or {}), **(other or {}))
    #
    #         self.preprocess_transition(transition, exploration=True)
    #         self.memory.store(transition)
    #         self.log(random_action_env=action_env, random_action=action, **debug)
    #
    #         if terminal:
    #             state = self.env.reset()
    #             state = self.preprocess(state, evaluation=False)
    #             print(f'Exploratory episode terminated. Steps left: {steps - 1}.')
    #
    #         steps -= 1

    # TODO: avoid duplicated code!
    def learn(self, episodes: int, timesteps: int, render: Union[bool, int, None] = False, should_close=True,
              evaluation: Union[dict, bool] = None, exploration_steps=0, save=True):
        """Training loop"""
        assert episodes > 0
        assert timesteps > 0

        total_seconds = 0.0
        self.on_start(episodes, timesteps)

        # Render:
        if render is True:
            render_freq = 1  # render each episode

        elif (render is False) or (render is None) or (render < 0):
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
            evaluation = {}
            eval_freq = episodes + 1  # never evaluate

        # Saving:
        if save:
            # track 'average return' to determine the best agent, also prefer newer agents (if equally good)
            best_return = self.config.get('eval_total_reward', -np.inf)
            should_save = True
        else:
            should_save = False

        # Exploration:
        exploration_steps = int(exploration_steps)
        is_exploring = exploration_steps > 0

        # Learning loop (outer):
        episode = 0

        while episode < episodes:
            episode += 1
            self.on_episode_start(episode, exploration=is_exploring)

            should_render = (not is_exploring) and (episode % render_freq == 0)
            should_evaluate = (not is_exploring) and (episode % eval_freq == 0)

            episode_reward = 0.0
            discount = 1.0
            discounted_reward = 0.0
            t0 = time.time()

            state = self.env.reset()
            state = self.preprocess(state)

            # inner-loop:
            t = 0
            while t < timesteps + 1:
                t += 1

                self.timestep = t
                self.total_steps += 1

                if is_exploring and exploration_steps <= 0:
                    # reset episode count, since we discard the episodes consumed for exploration
                    episode = 0
                    is_exploring = False

                    print('Exploration terminated.')
                    break
                else:
                    exploration_steps -= 1

                if should_render:
                    self.env.render()

                # Agent prediction
                if is_exploring:
                    # TODO: set `inference=True`?
                    action, other, debug = self.act_randomly(state)
                else:
                    action, other, debug = self.act(state, inference=True)

                action_env = self.convert_action(action)

                # Environment step
                for _ in range(self.repeat_action):
                    next_state, reward, terminal, info = self.env.step(action=action_env)

                    is_truncated = info.get('TimeLimit.truncated', False)
                    is_failure = terminal and not is_truncated  # if truncation occurs do not bootstrap

                    episode_reward += reward
                    discounted_reward += reward * discount
                    discount *= self.gamma

                    if terminal:
                        break

                # TODO: next_state not pre-processed, pre-process or avoid storing them!
                transition = dict(state=state, action=action, reward=reward, next_state=next_state,
                                  terminal=is_failure, **(info or {}), **(other or {}))

                self.on_transition(transition, terminal, exploration=is_exploring)

                if is_exploring:
                    self.log(random_action_env=action_env, random_action=action, **debug)
                else:
                    self.log_env(action=action_env, **debug)

                if terminal or (t == timesteps) or self.termination_condition(transition):
                    seconds = time.time() - t0
                    total_seconds += seconds

                    self.log(timestep=t, total_steps=self.total_steps, time_seconds=total_seconds,
                             seconds_per_epoch=seconds)

                    self.on_termination(last_transition=transition, exploration=is_exploring)

                    if is_exploring:
                        episode = 0
                        print(f'Exploration episode terminated. Steps left: {exploration_steps - 1}.')
                    else:
                        print(f'Episode {episode} terminated after {t} timesteps in {round(seconds, 3)}s ' +
                              f'with reward {round(episode_reward, 3)}')
                    break
                else:
                    state = next_state
                    state = self.preprocess(state)

            self.on_episode_end(episode, episode_reward, exploration=is_exploring)
            self.log(episode_reward_discounted=discounted_reward)

            if should_evaluate:
                eval_rewards = self.evaluate(**evaluation)
                mean_rewards = np.mean(eval_rewards).item()

                self.log(eval_rewards=eval_rewards)
                print(f'[Evaluation] average return: {round(mean_rewards, 2)}, '
                      f'std: {np.round(np.std(eval_rewards), 2)}')

                if should_save:
                    if np.floor(mean_rewards) >= best_return:
                        self.config['eval_total_reward'] = mean_rewards
                        self.save(folder=f'e{episode}-r{round(mean_rewards, 2)}')

                        best_return = mean_rewards
                        print(f'Saved [{round(best_return, 2)}]')

            if self.should_record:
                self.record(episode)

        self.is_learning = False
        self.on_close(should_close)

    # TODO: provide choice for evaluation criterion: mean, median, std...
    def evaluate(self, episodes: int, timesteps: int, render: Union[bool, int] = True,
                 should_close=False) -> np.ndarray:
        self.is_evaluating = True

        if render is True:
            render_freq = 1  # render each episode

        elif (render is False) or (render is None) or (render < 0):
            render_freq = episodes + 1  # never render
        else:
            render_freq = int(render)  # render at specified frequency

        episodic_rewards = np.zeros(shape=[episodes], dtype=np.float32)

        for episode in range(1, episodes + 1):
            self.on_episode_start(episode, evaluation=True)

            should_render = episode % render_freq == 0

            state = self.env.reset()
            state = self.preprocess(state, evaluation=True)

            t = 0
            while t < timesteps + 1:
                t += 1

                if should_render:
                    self.env.render()

                # Agent prediction
                action, other, debug = self.act(state, deterministic=True)
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

        self.is_evaluating = False
        return episodic_rewards

    @classmethod
    def test(cls, args: dict, network_summary=False, **kwargs):
        """Rapid testing"""
        agent = cls(**kwargs)

        if network_summary:
            agent.summary()
            breakpoint()

        agent.learn(**args)

    def termination_condition(self, transition: dict) -> bool:
        """Override to provide additional termination to trigger the end of an episode"""
        return False

    def preprocess(self, state, evaluation=False):
        if isinstance(state, dict):
            return {f'state_{k}': utils.to_tensor(v, expand_axis=0) for k, v in state.items()}

        return utils.to_tensor(state, expand_axis=0)

    def preprocess_reward(self, reward):
        self.log(reward_original=reward)

        # reward clipping
        r_min, r_max = self.reward_clip_range
        return np.clip(reward, a_min=r_min, a_max=r_max) * self.reward_scale()

    def preprocess_action(self, action):
        if isinstance(action, dict):
            return {f'action_{k}': v for k, v in action.items()}

        return action

    def preprocess_transition(self, transition: dict, exploration=False):
        transition['reward'] = self.preprocess_reward(transition['reward'])
        # transition['action'] = self.preprocess_action(transition['action'])

    # TODO: supports only one-depth nested dicts
    # TODO: what about tuples?
    def log(self, average=False, **kwargs):
        if self.should_log:
            data = dict(average=average)

            for key, value in kwargs.items():
                if isinstance(value, dict):
                    data.update({f'{key}__{k}': v for k, v in value.items()})
                else:
                    data[key] = value

            self.summary_queue.put(data)

    def log_transition(self, transition: dict):
        # action = transition['action']
        #
        # if isinstance(action, dict):
        #     self.log(reward=transition['reward'], **action)
        # else:
        #     self.log(reward=transition['reward'], action=action)

        self.log(reward=transition['reward'], **utils.to_dict_for_log(transition['action'], name='action'),
                 state_hist=transition['state'], action_hist=transition['action'])
        # self.log_action(action=transition['action'])

    def log_env(self, action, **kwargs):
        # if isinstance(action, dict):
        #     actions_env = {f'action_env_{k}': v for k, v in action.items()}
        #     self.log(**actions_env, **kwargs)
        # else:
        #     self.log(action_env=action, **kwargs)

        self.log(**kwargs, **utils.to_dict_for_log(action, name='action_env'))
        # self.log_action(action, prefix='_env')

    # def log_action(self, action: Union[dict, int, float, np.ndarray, tf.Tensor], prefix=''):
    #     # scalar (single) action
    #     if utils.is_scalar(action):
    #         return self.log(**{f'action{prefix}': action})
    #
    #     # multiple scalar actions
    #     if isinstance(action, np.ndarray) or tf.is_tensor(action):
    #         if len(action.shape) == 1:
    #             d = {f'action{prefix}_{i}': a for i, a in enumerate(action)}
    #         else:
    #             d = {f'action{prefix}_{i}': np.mean(action[:, i]) for i in range(action.shape[-1])}
    #
    #         return self.log(**d)
    #
    #     # composite action
    #     assert isinstance(action, dict)
    #     d = {}
    #
    #     for k, v in action.items():
    #         if utils.is_scalar(v):
    #             d[f'action{prefix}-{k}'] = v
    #
    #         if isinstance(v, np.ndarray) or tf.is_tensor(v):
    #             if len(v.shape) == 1:
    #                 d.update({f'action{prefix}_{i}': a for i, a in enumerate(v)})
    #             else:
    #                 d.update({f'action{prefix}_{i}': np.mean(v[:, i]) for i in range(v.shape[-1])})
    #
    #     self.log(**d)

    def log_dynamic_parameters(self):
        self.log(**({k: p.value for k, p in self.dynamic_parameters.items()}))

    def summary(self):
        """Networks summary"""
        for network in self.networks.values():
            network.summary()

    def update_config(self, **kwargs):
        """Stores the given variables in the configuration dict for later saving"""
        for k, v in kwargs.items():
            self.config[k] = v

    def load_config(self, path: str):
        # load configuration file first
        with open(os.path.join(path, 'config.json'), 'r') as file:
            self.config = json.load(file)
            print('config loaded.')

        # then update the dynamic parameters according to the config file
        for name, param in self.dynamic_parameters.items():
            param.load(config=self.config[name])

        print('dynamic parameters loaded.')

    def save_config(self, path: str):
        self.update_config(**{k: p.get_config() for k, p in self.dynamic_parameters.items()})

        with open(os.path.join(path, 'config.json'), 'w') as file:
            json.dump(self.config, fp=file, indent=3)
            print('config saved.')

    def reset(self):
        pass

    def load(self, path: str = None):
        """Loads the past agent's state (weights).
            - If `path` is None: the agent with the highest rewards under the path:
              "weights/name/<timestamp>/<episode><reward>" is loaded by default."""
        if path is None:
            # find the path associated with the best trained agent, so far.
            base_path = os.path.join(self.save_dir, self.name)
            run_paths = [os.path.join(base_path, timestamp) for timestamp in os.listdir(base_path)]

            # flatten `runs_path` to get each evaluation episode
            paths = []

            for run_path in run_paths:
                eval_paths = [(run_path, folder) for folder in os.listdir(run_path)]
                paths.extend(eval_paths)

            # split by total reward "r"
            paths = map(lambda x: (os.path.join(*x), x[-1].split('r')[0][1:-1], x[-1].split('r')[-1]),
                        paths)  # [(path, episode, reward)]

            # sort by total reward
            paths = sorted(paths, key=lambda k: [float(k[-1]), float(k[-2])])  # k = (reward, episode)

            # take the best path (the last); the agent will be loaded at such path
            path = paths[-1][0]
        else:
            assert isinstance(path, str)

        self.load_config(path)
        self.load_weights(path)
        print(f'Loaded from path "{path}"; with total reward {round(self.config["eval_total_reward"], 2)}')

    def save(self, folder: str):
        """Saves the agent's state"""
        save_path = os.path.join(self.base_path, folder)

        self.save_weights(path=save_path)
        self.save_config(path=save_path)

    def load_weights(self, path: str):
        for name, network in self.networks.items():
            network.load_weights(filepath=os.path.join(path, name), by_name=False)

    def save_weights(self, path: str):
        for name, network in self.networks.items():
            network.save_weights(filepath=os.path.join(path, name))

    def get_weights(self) -> dict:
        """Returns a dictionary of network -> weights"""
        return {k: network.get_weights() for k, network in self.networks.items()}

    def set_weights(self, agent: 'Agent' = None, weights=None):
        """Sets the weights for the current agent's networks given either another agent instance or a dict of weights"""
        if agent is not None:
            weights = agent.get_weights()

        assert isinstance(weights, dict)
        assert weights.keys() == self.networks.keys()

        for k, w in weights.items():
            self.networks[k].set_weights(w)

    def on_episode_start(self, episode: int, evaluation=False, exploration=False):
        """Called at the beginning of each episode"""
        self.episode = episode
        self.timestep = 0
        self.reset()

    def on_episode_end(self, episode: int, episode_reward: float, evaluation=False, exploration=False):
        """Called *after* the end of each episode.
            - Used for memory-specific stuff.
            - Or, to update the agent (e.g. PPO)
        """
        if exploration:
            return

        if not evaluation:
            for parameter in self.dynamic_parameters.values():
                parameter.on_episode()

            self.log_dynamic_parameters()
            self.log(episode_reward=episode_reward)

    # TODO: start/end ?
    def on_timestep(self, timestep: int):
        pass

    def on_transition(self, transition: dict, terminal: bool, exploration=False):
        """Called upon a transition occurs (agent prediction + environment step).
            - Useful for logging or pre-processing the transition.
            - Also for storing transition in memory, also conditional on actual timestep/episode num.
            - Or, updating the agent (e.g. DQN)
        """
        self.preprocess_transition(transition, exploration=exploration)
        self.log_transition(transition)
        self.memory.store(transition)

    # def on_log(self, *args, **kwargs):
    #     """Called *after* environment step, usually used to log actions"""
    #     pass

    def on_termination(self, last_transition, exploration=False):
        """Called *exactly* at the end of each episode, due to terminal state or maximum number of timesteps reached.
            - Used for agent-specific memory stuff.
            - Or, to update the agent (e.g. PPO)
        """
        pass

    def on_start(self, episodes: int, timesteps: int):
        """Called *before* the training loop commence"""
        self.is_learning = True
        self.total_steps = 0
        self.max_timesteps = timesteps

        if self.should_log and not self.statistics.is_alive():
            self.statistics.start()

    def on_close(self, should_close: bool):
        """Called *after* the training loop ends"""
        self.is_learning = False
        self.timestep = 0
        self.episode = 0

        self.close_summary()

        if should_close:
            print('Closing env...')
            self.env.close()

    def close_summary(self):
        """Closes the SummaryProcess"""
        if self.should_log:
            print('Writing summaries...')
            self.summary_stop_event.set()
            self.statistics.close()


# class ParallelAgent(Agent):
#     """Base class for Agents that uses parallel environments (e.g. A2C, PPO, ...)"""
#     def __init__(self, env, *args, num_actors: int, name='parallel-agent', **kwargs):
#         assert num_actors >= 1
#
#         self.num_actors = int(num_actors)
#         # self.max_timesteps = 0  # being init in `self.learn(...)`
#
#         super().__init__(env=ParallelEnv(env, num=self.num_actors), *args, name=name, **kwargs)
#
#     def _init_action_space(self):
#         action_space = self.env.action_space
#
#         if isinstance(action_space, gym.spaces.Box):
#             self.num_actions = action_space.shape[0]
#
#             # continuous:
#             if action_space.is_bounded():
#                 self.distribution_type = 'beta'
#
#                 self.action_low = tf.constant(action_space.low, dtype=tf.float32)
#                 self.action_high = tf.constant(action_space.high, dtype=tf.float32)
#                 self.action_range = tf.constant(action_space.high - action_space.low, dtype=tf.float32)
#
#                 def convert_action(actions) -> list:
#                     return [tf.squeeze(a * self.action_range + self.action_low).numpy() for a in actions]
#
#                 self.convert_action = convert_action
#             else:
#                 self.distribution_type = 'gaussian'
#                 self.convert_action = lambda actions: [tf.squeeze(a).numpy() for a in actions]
#         else:
#             # discrete:
#             assert isinstance(action_space, gym.spaces.Discrete)
#             self.distribution_type = 'categorical'
#
#             self.num_actions = 1
#             self.num_classes = action_space.n
#             self.convert_action = lambda actions: [tf.cast(tf.squeeze(a), dtype=tf.int32).numpy() for a in actions]
#
#     def act(self, states, deterministic=False, inference=False, **kwargs) -> Tuple[tf.Tensor, dict, dict]:
#         raise NotImplementedError
#
#     # TODO: `evaluation` and `exploration`
#     def learn(self, episodes: int, timesteps: int, render: Union[bool, int, None] = False, should_close=True,
#               evaluation: Union[dict, bool] = None, exploration_steps=0, save=True):
#         """Training loop"""
#         assert episodes > 0
#         assert timesteps > 0
#
#         t0 = time.time()
#         tot_rew = 0.0
#
#         # Render:
#         if render is True:
#             render_freq = 1  # render each episode
#
#         elif render is False or render is None:
#             render_freq = episodes + 1  # never render
#         else:
#             render_freq = int(render)  # render at specified frequency
#
#         # # Evaluation:
#         # if isinstance(evaluation, dict):
#         #     eval_freq = evaluation.pop('freq', episodes + 1)  # default: never evaluate
#         #     assert isinstance(eval_freq, int)
#         #
#         #     evaluation['should_close'] = False
#         #     evaluation.setdefault('episodes', 1)  # default: evaluate on just 1 episode
#         #     evaluation.setdefault('timesteps', timesteps)  # default: evaluate on the same number of timesteps
#         #     evaluation.setdefault('render', render)  # default: same rendering options
#         # else:
#         #     eval_freq = episodes + 1  # never evaluate
#
#         # # Saving:
#         # if save:
#         #     # track 'average return' to determine best agent, also prefer newer agents (if equally good)
#         #     best_return = -2 ** 32
#         #     should_save = True
#         # else:
#         #     should_save = False
#
#         # # Exploration:
#         # self.explore(steps=int(exploration_steps))
#
#         self.on_start(episodes, timesteps)
#         # self.max_timesteps = timesteps
#
#         # Learning-loop:
#         for episode in range(1, episodes + 1):
#             self.on_episode_start(episode)
#
#             should_render = episode % render_freq == 0
#             # should_evaluate = episode % eval_freq == 0
#
#             episode_reward = 0.0
#             t0 = time.time()
#
#             states = self.env.reset()
#             states = self.preprocess(states)
#
#             # for t in range(1, timesteps + 1):
#             t = 0
#             while t < timesteps + 1:
#                 t += 1
#                 self.timestep = t
#                 self.total_steps += 1
#
#                 if should_render:
#                     self.env.render()
#
#                 # Agent prediction
#                 actions, other, debug = self.act(states)
#                 actions_env = self.convert_action(actions)
#
#                 # Environment step
#                 for _ in range(self.repeat_action):
#                     next_states, rewards, terminals, info = self.env.step(action=actions_env)
#                     episode_reward += np.mean(rewards)
#
#                     if any(terminals):
#                         break
#
#                 transition = dict(state=states, action=actions, reward=rewards, next_state=next_states,
#                                   terminal=terminals, **(info or {}), **(other or {}))
#
#                 self.on_transition(transition, terminals)
#                 self.log_env(action=actions_env, **debug)
#
#                 if any(terminals) or (t == timesteps) or self.termination_condition(transition):
#                     print(f'Episode {episode} terminated after {t} timesteps in {round((time.time() - t0), 3)}s ' +
#                           f'with reward {round(episode_reward, 3)}')
#                     tot_rew += episode_reward
#
#                     self.log(timestep=t, total_steps=self.total_steps)
#                     self.on_termination(last_transition=transition)
#                     break
#                 else:
#                     states = next_states
#                     states = self.preprocess(states)
#
#             self.on_episode_end(episode, episode_reward)
#
#             # if should_evaluate:
#             #     eval_rewards = self.evaluate(**evaluation)
#             #
#             #     self.log(eval_rewards=eval_rewards)
#             #     print(f'[Evaluation] average return: {np.round(np.mean(eval_rewards), 2)}, '
#             #           f'std: {np.round(np.std(eval_rewards), 2)}')
#             #
#             #     if should_save:
#             #         average_return = np.floor(np.mean(eval_rewards))
#             #
#             #         # TODO: when saving also save 'best_return' into the agent's config
#             #         if average_return >= best_return:
#             #             self.save()
#             #             best_return = average_return
#             #             print(f'Saved [{round(best_return, 2)}]')
#
#             if self.should_record:
#                 self.record(episode)
#
#         print(f'Time taken {round(time.time() - t0, 3)}s.')
#         print(f'Total episodic reward: {round(tot_rew, 3)}')
#         self.on_close(should_close)
#
#     def preprocess(self, states, evaluation=False):
#         if isinstance(states[0], dict):
#             # states_ = {f'state_{k}': [v] for k, v in states[0].items()}
#             states_ = {f'state_{k}': [] for k in states[0].keys()}
#
#             for state in states:
#                 for k, v in state.items():
#                     states_[f'state_{k}'].append(v)
#
#             return states_
#
#         return states
#
#     def log_transition(self, transition: Dict[str, list]):
#         data = dict()
#
#         for i, (reward, action) in enumerate(zip(transition['reward'], transition['action'])):
#             data[f'reward_{i}'] = reward
#
#             if isinstance(action, dict):
#                 for k, v in action.items():
#                     data[f'action_{i}_{k}'] = v
#             else:
#                 data[f'action_{i}'] = action
#
#         self.log(**data)
#
#     # TODO: generalize to `log_actions`?
#     def log_env(self, action: list, **kwargs):
#         actions = action
#         data = dict()
#
#         for i, action in enumerate(actions):
#             if isinstance(action, dict):
#                 for k, v in action.items():
#                     data[f'action_env_{k}_{i}'] = v
#             else:
#                 data[f'action_env_{i}'] = action
#
#         self.log(**data, **kwargs)


# class ParallelMemory:
#     def __init__(self, memories: List[Memory]):
#         assert len(memories) >= 1
#         self.memories = memories
#
#     def __len__(self):
#         return sum(memory.index for memory in self.memories)
#
#     def full_enough(self, amount: int) -> bool:
#         count = [memory.index for memory in self.memories]
#         return np.sum(count) >= amount
#
#     def all_full(self) -> bool:
#         return all(memory.is_full() for memory in self.memories)
#
#     def any_full(self) -> bool:
#         return any(memory.is_full() for memory in self.memories)
#
#     def all_full_enough(self, amount: int) -> bool:
#         amount = int(amount)
#         return all(memory.full_enough(amount) for memory in self.memories)
#
#     def any_full_enough(self, amount: int) -> bool:
#         amount = int(amount)
#         return any(memory.full_enough(amount) for memory in self.memories)
#
#     def store(self, transition: Dict[str, np.ndarray]):
#         keys = transition.keys()
#
#         # unpack `transition` to get a list of per-agent tuples
#         for i, tuple_ in enumerate(zip(*transition.values())):
#             self.memories[i].store(transition={k: v for k, v in zip(keys, tuple_)})
#
#     def clear(self):
#         for memory in self.memories:
#             memory.clear()
#
#     def on_update(self, *args, **kwargs):
#         for memory in self.memories:
#             memory.on_update(*args, **kwargs)
#
#     def end_trajectory(self, last_values: tf.Tensor) -> dict:
#         values = tf.unstack(last_values)
#         debug = {}
#
#         for memory, value in zip(self.memories, values):
#             d = memory.end_trajectory(last_value=value.numpy())
#
#             for k, v in d.items():
#                 if k in debug:
#                     debug[k].append(v)
#                 else:
#                     debug[k] = [v]
#
#         return debug
#
#     def get_data(self) -> Dict[str, np.ndarray]:
#         mem_data = [memory.get_data() for memory in self.memories]
#
#         # concat all `data`
#         data = {k: [] for k in mem_data[0].keys()}
#
#         for datum in mem_data:
#             for k, v in datum.items():
#                 data[k].append(v)
#
#         return {k: np.concatenate(v, axis=0) for k, v in data.items()}
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
#         # batch, shuffle, repeat, prefetch
#         ds = ds.batch(batch_size)
#         # ds = ds.shuffle(buffer_size=batch_size * 2, reshuffle_each_iteration=True)
#         ds = ds.repeat(count=repeat)
#
#         return ds.prefetch(buffer_size=2)
#
#     def update_warning(self, batch_size: int):
#         self.memories[0].update_warning(batch_size=int(batch_size))


class ParallelAgent(Agent):
    """Base class for Agents that uses parallel environments (e.g. A2C, PPO, ...)"""
    def __init__(self, env, *args, num_actors: int, name='parallel_agent', parallel_env=SequentialEnv, **kwargs):
        assert num_actors >= 1
        self.num_actors = int(num_actors)

        seed = kwargs.get('seed', utils.GLOBAL_SEED)
        env_kwargs = kwargs.pop('env_kwargs', {})

        super().__init__(*args, env=parallel_env(env, num=self.num_actors, seed=seed, **env_kwargs),
                         name=name, **kwargs)

        assert isinstance(self.env, AbstractParallelEnv)

    # @property
    # def memory(self) -> ParallelMemory:
    #     if self._memory is None:
    #         memories = [self.define_memory() for _ in range(self.num_actors)]
    #         self._memory = ParallelMemory(memories)
    #
    #     return self._memory

    def _init_action_space(self):
        action_space = self.env.action_space

        if isinstance(action_space, gym.spaces.Box):
            self.num_actions = action_space.shape[0]

            # continuous:
            if action_space.is_bounded():
                self.action_low = tf.constant(action_space.low, dtype=tf.float32)
                self.action_high = tf.constant(action_space.high, dtype=tf.float32)
                self.action_range = tf.constant(action_space.high - action_space.low, dtype=tf.float32)

                def convert_action(actions) -> list:
                    return [tf.squeeze(a * self.action_range + self.action_low).numpy() for a in actions]

                self.convert_action = convert_action
            else:
                self.convert_action = lambda actions: [tf.squeeze(a).numpy() for a in actions]
        else:
            # discrete:
            assert isinstance(action_space, gym.spaces.Discrete)

            self.num_actions = 1
            self.num_classes = action_space.n
            self.convert_action = lambda actions: [tf.cast(tf.squeeze(a), dtype=tf.int32).numpy() for a in actions]

    def act(self, states, deterministic=False, inference=False, **kwargs) -> Tuple[tf.Tensor, dict, dict]:
        raise NotImplementedError

    def act_evaluation(self, states, **kwargs):
        raise NotImplementedError

    # # TODO: implement "saving" and "exploration"
    # def learn(self, episodes: int, timesteps: int, render: Union[bool, int, None] = False, should_close=True,
    #           evaluation: Union[dict, bool] = None, exploration_steps=0, save=True):
    #     """Training loop"""
    #     assert episodes > 0
    #     assert timesteps > 0
    #
    #     t0 = time.time()
    #     tot_rew = 0.0
    #
    #     # Render:
    #     if render is True:
    #         render_freq = 1  # render each episode
    #
    #     elif render is False or render is None:
    #         render_freq = episodes + 1  # never render
    #     else:
    #         render_freq = int(render)  # render at specified frequency
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
    #     # Learning-loop:
    #     for episode in range(1, episodes + 1):
    #         self.on_episode_start(episode)
    #
    #         should_render = episode % render_freq == 0
    #         episode_reward = 0.0
    #         discounted_reward = 0.0
    #         discount = 1.0
    #         t0 = time.time()
    #
    #         states = self.env.reset()
    #         states = self.preprocess(states)
    #
    #         # inner-loop
    #         t = 0
    #
    #         while t < timesteps + 1:
    #             t += 1
    #             self.timestep = t
    #             self.total_steps += 1
    #
    #             if should_render:
    #                 self.env.render()
    #
    #             # Agent prediction
    #             actions, other, debug = self.act(states)
    #             actions_env = self.convert_action(actions)
    #
    #             # Environment step
    #             for _ in range(self.repeat_action):
    #                 next_states, rewards, terminals, info = self.env.step(action=actions_env)
    #
    #                 is_truncated = [x.get('TimeLimit.truncated', False) for x in info]
    #                 # is_truncated = info.get('TimeLimit.truncated', np.full_like(terminals, fill_value=False))
    #                 is_failure = np.logical_and(terminals, np.logical_not(is_truncated))
    #
    #                 episode_reward += np.mean(rewards)
    #                 discounted_reward += np.mean(rewards) * discount
    #                 discount *= self.gamma
    #
    #                 if any(terminals):
    #                     break
    #
    #             transition = dict(state=states, action=actions, reward=rewards, next_state=next_states,
    #                               terminal=is_failure, **(info or {}), **(other or {}))
    #
    #             self.on_transition(transition, terminals)
    #             self.log_env(action=actions_env, **debug)
    #
    #             if any(terminals) or (t == timesteps) or self.termination_condition(transition):
    #                 print(f'Episode {episode} terminated after {t} timesteps in {round((time.time() - t0), 3)}s ' +
    #                       f'with reward {round(episode_reward, 3)}')
    #                 tot_rew += episode_reward
    #
    #                 self.log(timestep=t, total_steps=self.total_steps)
    #                 self.on_termination(last_transition=transition)
    #                 break
    #             else:
    #                 states = next_states
    #                 states = self.preprocess(states)
    #
    #         self.on_episode_end(episode, episode_reward)
    #         self.log(episode_reward_discounted=discounted_reward)
    #
    #         if episode % eval_freq == 0:
    #             eval_rewards = self.evaluate(**evaluation)
    #
    #             self.log(eval_rewards=eval_rewards)
    #             print(f'[Evaluation] average return: {np.round(np.mean(eval_rewards), 2)}, '
    #                   f'std: {np.round(np.std(eval_rewards), 2)}')
    #
    #             # if should_sav:
    #             #     average_return = np.floor(np.mean(eval_rewards))
    #             #
    #             #     # TODO: when saving also save 'best_return' into the agent's config
    #             #     if average_return >= best_return:
    #             #         self.save()
    #             #         best_return = average_return
    #             #         print(f'Saved [{round(best_return, 2)}]')
    #
    #         if self.should_record:
    #             self.record(episode)
    #
    #     print(f'Time taken {round(time.time() - t0, 3)}s.')
    #     print(f'Total episodic reward: {round(tot_rew, 3)}')
    #     self.on_close(should_close)

    # def evaluate(self, episodes: int, timesteps: int, render: Union[bool, int] = True,
    #              should_close=False) -> np.ndarray:
    #     self.is_evaluating = True
    #
    #     if render is True:
    #         render_freq = 1  # render each episode
    #
    #     elif render is False or render is None:
    #         render_freq = episodes + 1  # never render
    #     else:
    #         render_freq = int(render)  # render at specified frequency
    #
    #     episodic_rewards = np.zeros(shape=[episodes], dtype=np.float32)
    #
    #     if hasattr(self.env, 'envs'):
    #         env: gym.Env = self.env.envs[0]  # evaluation will be performed on just one environment
    #     else:
    #         env: gym.Env = self.env.make_env(which=self.env.env_name, rank=0)
    #         should_close = True
    #
    #     for episode in range(1, episodes + 1):
    #         self.on_episode_start(episode, evaluation=True)
    #
    #         should_render = episode % render_freq == 0
    #
    #         state = env.reset()
    #         state = self.preprocess([state], evaluation=True)
    #
    #         for t in range(1, timesteps + 1):
    #             if should_render:
    #                 env.render()
    #
    #             # Agent prediction
    #             action, other, debug = self.act(state, deterministic=True)
    #             action_env = self.convert_action(action)[0]
    #
    #             # Environment step
    #             for _ in range(self.repeat_action):
    #                 next_state, reward, terminal, info = env.step(action=action_env)
    #                 episodic_rewards[episode - 1] += reward
    #
    #                 if terminal:
    #                     break
    #
    #             if terminal or (t == timesteps):
    #                 print(f'Evaluation episode {episode}/{episodes} terminated after {t}/{timesteps} timesteps ' +
    #                       f'with reward {round(episodic_rewards[episode - 1], 3)}.')
    #                 break
    #             else:
    #                 state = next_state
    #                 state = self.preprocess([state], evaluation=True)
    #
    #         self.on_episode_end(episode, episode_reward=episodic_rewards[episode - 1], evaluation=True)
    #
    #     if should_close:
    #         env.close()
    #
    #     self.is_evaluating = False
    #     return episodic_rewards

    # TODO: missing "frame-skip"
    def learn(self, episodes: int, timesteps: int, should_close=True, evaluation: Union[dict, bool] = None,
              exploration_steps=0, save=True, **kwargs):
        assert episodes > 0
        assert timesteps > 0

        total_seconds = 0
        self.on_start(episodes, timesteps)

        # init evaluation args:
        if isinstance(evaluation, dict):
            eval_freq = evaluation.pop('freq', episodes + 1)  # default: never evaluate
            assert isinstance(eval_freq, int)

            evaluation.setdefault('episodes', 1)  # default: evaluate on just 1 episode
            evaluation.setdefault('timesteps', timesteps)  # default: evaluate on the same number of timesteps
            evaluation.setdefault('render', False)  # default: do not render
        else:
            evaluation = {}
            eval_freq = episodes + 1  # never evaluate

        # Saving:
        if save:
            # track 'average return' to determine the best agent, also prefer newer agents (if equally good)
            best_return = self.config.get('eval_total_reward', -np.inf)
            should_save = True
        else:
            should_save = False

        # Exploration:
        exploration_steps = int(exploration_steps)
        is_exploring = exploration_steps > 0

        # Learning loop (outer):
        episode = 0

        while episode < episodes:
            episode += 1
            self.on_episode_start(episode, exploration=is_exploring)

            episode_reward = 0.0
            discounted_reward = 0.0
            discount = 1.0
            t0 = time.time()

            states = self.env.reset()
            states = self.preprocess(states)

            # inner-loop:
            t = 0
            while t < timesteps + 1:
                t += 1

                self.timestep = t
                self.total_steps += 1

                if is_exploring and exploration_steps <= 0:
                    # reset episode count, since we discard the episodes consumed for exploration
                    episode = 0
                    is_exploring = False

                    print('Exploration terminated.')
                    break
                else:
                    exploration_steps -= 1

                # Agent prediction
                if is_exploring:
                    actions, other, debug = self.act_randomly(states)
                else:
                    actions, other, debug = self.act(states, inference=True)

                actions_env = self.convert_action(actions)

                # Environment step
                next_states, rewards, terminals, info = self.env.step(action=actions_env)

                is_truncated = [x.get('TimeLimit.truncated', False) for x in info.values()]
                is_failure = np.logical_and(terminals, np.logical_not(is_truncated))

                episode_reward += np.mean(rewards)
                discounted_reward += np.mean(rewards) * discount
                discount *= self.gamma

                transition = dict(state=states, action=actions, reward=rewards, next_state=next_states,
                                  terminal=is_failure, info=info, **(other or {}))

                self.on_transition(transition, terminals, exploration=is_exploring)

                if is_exploring:
                    self.log(random_actions_env=actions_env, random_actions=actions, **debug)
                else:
                    self.log_env(action=actions_env, **debug)

                if (t == timesteps) or self.memory.is_full() or self.termination_condition(transition):
                    seconds = time.time() - t0
                    total_seconds += seconds

                    self.log(timestep=t, total_steps=self.total_steps, time_seconds=total_seconds,
                             seconds_per_epoch=seconds)
                    self.on_termination(last_transition=transition, exploration=is_exploring)

                    if is_exploring:
                        episode = 0
                        print(f'Exploration episode terminated. Steps left: {exploration_steps - 1}.')
                    else:
                        print(f'Episode {episode} terminated after {t} timesteps in {round(seconds, 3)}s ' +
                              f'with reward {round(episode_reward, 3)}')
                    break
                else:
                    # if any(terminals):
                    #     reset_states = self.env.reset(terminating=terminals)
                    #
                    #     for idx, state in reset_states:
                    #         next_states[idx] = state

                    states = next_states
                    states = self.preprocess(states)

            self.on_episode_end(episode, episode_reward, exploration=is_exploring)

            # Evaluate
            if (not is_exploring) and (episode % eval_freq == 0):
                eval_rewards = self.evaluate(**evaluation)
                mean_rewards = np.mean(eval_rewards).item()

                self.log(eval_rewards=eval_rewards)
                print(f'[Evaluation] average return: {round(mean_rewards, 2)}, '
                      f'std: {np.round(np.std(eval_rewards), 2)}')

                if should_save:
                    if np.floor(mean_rewards) >= best_return:
                        self.config['eval_total_reward'] = mean_rewards
                        self.save(folder=f'e{episode}-r{round(mean_rewards, 2)}')

                        best_return = mean_rewards
                        print(f'Saved [{round(best_return, 2)}]')

        self.is_learning = False
        self.on_close(should_close)

    def evaluate(self, episodes: int, timesteps: int, render: Union[bool, int] = True, **kwargs) -> np.ndarray:
        self.is_evaluating = True

        if render is True:
            render_freq = 1  # render each episode

        elif (render is False) or (render is None) or (render < 0):
            render_freq = episodes + 1  # never render
        else:
            render_freq = int(render)  # render at specified frequency

        episodic_rewards = np.zeros(shape=[episodes], dtype=np.float32)

        # evaluation environment
        env: gym.Env = self.env.get_evaluation_env()
        env.seed(seed=self.seed)

        for episode in range(1, episodes + 1):
            self.on_episode_start(episode, evaluation=True)

            should_render = episode % render_freq == 0

            state = env.reset()
            state = self.preprocess([state], evaluation=True)

            t = 0
            while t < timesteps + 1:
                t += 1

                if should_render:
                    env.render()

                # Agent prediction
                # action, other, debug = self.act(state, deterministic=True)
                action = self.act_evaluation(state)
                action_env = self.convert_action(action)[0]

                # Environment step
                for _ in range(self.repeat_action):
                    next_state, reward, terminal, info = env.step(action=action_env)
                    episodic_rewards[episode - 1] += reward

                    if terminal:
                        break

                if terminal or (t == timesteps):
                    print(f'Evaluation episode {episode}/{episodes} terminated after {t}/{timesteps} timesteps ' +
                          f'with reward {round(episodic_rewards[episode - 1], 3)}.')
                    break
                else:
                    state = next_state
                    state = self.preprocess([state], evaluation=True)

            self.on_episode_end(episode, episode_reward=episodic_rewards[episode - 1], evaluation=True)

        env.close()
        self.is_evaluating = False

        return episodic_rewards

    # def evaluate2(self, episodes: int, timesteps: int, render: Union[bool, int] = True,
    #               should_close=False) -> np.ndarray:
    #     self.is_evaluating = True
    #
    #     if render is True:
    #         render_freq = 1  # render each episode
    #
    #     elif render is False or render is None:
    #         render_freq = episodes + 1  # never render
    #     else:
    #         render_freq = int(render)  # render at specified frequency
    #
    #     episodic_rewards = np.zeros(shape=[episodes], dtype=np.float32)
    #
    #     for episode in range(1, episodes + 1):
    #         self.on_episode_start(episode, evaluation=True)
    #
    #         should_render = episode % render_freq == 0
    #
    #         states = self.env.reset()
    #         states = self.preprocess(states, evaluation=True)
    #
    #         for t in range(1, timesteps + 1):
    #             if should_render:
    #                 self.env.render()
    #
    #             # Agent prediction
    #             actions, other, debug = self.act(states, deterministic=True)
    #             actions_env = self.convert_action(actions)
    #
    #             # Environment step
    #             for _ in range(self.repeat_action):
    #                 next_states, rewards, terminals, info = self.env.step(action=actions_env)
    #                 episodic_rewards[episode - 1] += np.mean(rewards)
    #
    #                 if any(terminals):
    #                     break
    #
    #             if any(terminals) or (t == timesteps):
    #                 print(f'Evaluation episode {episode}/{episodes} terminated after {t}/{timesteps} timesteps ' +
    #                       f'with "average" reward {round(episodic_rewards[episode - 1], 3)}.')
    #                 break
    #             else:
    #                 states = next_states
    #                 states = self.preprocess(states, evaluation=True)
    #
    #         self.on_episode_end(episode, episode_reward=episodic_rewards[episode - 1], evaluation=True)
    #
    #     if should_close:
    #         self.env.close()
    #
    #     self.is_evaluating = False
    #     return episodic_rewards

    def preprocess(self, states: list, evaluation=False) -> list:
        if isinstance(states[0], dict):
            return [{f'state_{k}': v for k, v in s.items()} for s in states]

        elif evaluation:
            return tf.reshape(states[0], shape=(1,) + states[0].shape)

        return states

    # def preprocess_action(self, actions: list):
    #     if isinstance(actions[0], dict):
    #         return [{f'action_{k}': v for k, v in a.items()} for a in actions]
    #
    #     return actions

    # TODO: remove
    def stack_states(self, states: list):
        """Stacks states such that: (num_actors, |S|)"""
        if isinstance(states[0], dict):
            state = {k: [] for k in states[0].keys()}

            for s in states:
                for k, v in s.items():
                    state[k].append(v)

            return {k: tf.stack(v, axis=0) for k, v in state.items()}

        return tf.stack(states, axis=0)

    def record(self, timesteps: int, folder='video', seed=None, rename=True, **kwargs) -> Tuple[float, str]:
        from gym.wrappers import Monitor

        env: gym.Env = self.env.get_evaluation_env()
        env.seed(seed=seed or self.seed)

        timestamp = utils.actual_datetime()
        record_path = os.path.join(folder, self.name, timestamp)

        env = Monitor(env, directory=record_path, **kwargs)
        episode_reward = 0.0

        # start recording
        state = env.reset()
        state = self.preprocess([state], evaluation=True)

        t = 0
        while t < timesteps + 1:
            t += 1

            # Agent prediction
            action = self.act_evaluation(state)
            action_env = self.convert_action(action)[0]

            # Environment step
            for _ in range(self.repeat_action):
                next_state, reward, terminal, info = env.step(action=action_env)
                episode_reward += reward

                if terminal:
                    break

            if terminal or (t == timesteps):
                print(f'Record episode terminated after {t}/{timesteps} timesteps ' +
                      f'with reward {round(episode_reward, 3)}.')
                break
            else:
                state = next_state
                state = self.preprocess([state], evaluation=True)

        env.close()

        if rename:
            new_path = os.path.join(folder, self.name, f'{timestamp}-r{round(episode_reward, 2)}')

            utils.copy_folder(src=record_path, dst=new_path)
            utils.remove_folder(path=record_path)

            record_path = new_path

        return episode_reward, record_path

    # def log_transition(self, transition: Dict[str, list]):
    #     data = dict()
    #
    #     for i, (reward, action) in enumerate(zip(transition['reward'], transition['action'])):
    #         data[f'reward_{i}'] = reward
    #
    #         if isinstance(action, dict):
    #             for k, v in action.items():
    #                 data[f'action_{i}_{k}'] = v
    #         else:
    #             data[f'action_{i}'] = action
    #
    #     self.log(**data)

    def log_transition(self, transition: Dict[str, list]):
        self.log(reward=np.mean(transition['reward']), action=np.mean(transition['action']))

    # def log_env(self, action: list, **kwargs):
    #     actions = action
    #     data = dict()
    #
    #     for i, action in enumerate(actions):
    #         if isinstance(action, dict):
    #             for k, v in action.items():
    #                 data[f'action_env_{k}_{i}'] = v
    #         else:
    #             data[f'action_env_{i}'] = action
    #
    #     self.log(**data, **kwargs)

    def log_env(self, action: list, **kwargs):
        if isinstance(action[0], dict):
            # from List[dict] to dict[list]
            actions = {k: np.empty(shape=(len(action),) + v.shape) for k, v in action[0].items()}

            for i, a in enumerate(action):
                for k, v in a.items():
                    actions[k][i] = v

            self.log(**{f'action_{k}_env': np.mean(v) for k, v in actions.items()}, **kwargs)
        else:
            self.log(action_env=np.mean(action), **kwargs)


class RandomAgent(Agent):
    """Baseline agent that outputs random actions regardless the input state"""

    def __init__(self, *args, name='random-agent', **kwargs):
        kwargs['batch_size'] = 1
        kwargs['use_summary'] = False

        super().__init__(*args, name=name, **kwargs)

    def _init_action_space(self):
        action_space = self.env.action_space
        self.action_space: gym.Space = action_space

        assert isinstance(action_space, gym.Space)
        self.convert_action = lambda a: a

    def sample(self, *args, **kwargs):
        return self.action_space.sample()

    def act(self, state, **kwargs) -> Tuple[tf.Tensor, dict, dict]:
        return self.sample(state), {}, {}

    def learn(self, *args, **kwargs):
        raise RuntimeError('RandomAgent does not support learning.')
