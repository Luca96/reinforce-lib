import os
import gym
import json
import random
import numpy as np
import tensorflow as tf

from rl import utils
from typing import List, Dict, Union
from tensorflow.keras import layers


# TODO: implement RandomAgent
# TODO: actor-critic agent interface (to include policy/value network as well as loading/saving)?
# TODO: save agent configuration as json
class Agent:
    """Agent abstract class"""
    def __init__(self, env: Union[gym.Env, str], batch_size: int, seed=None, weights_dir='weights', name='agent',
                 log_mode='summary', drop_batch_reminder=False, skip_data=0, consider_obs_every=1,
                 shuffle_batches=False):
        if isinstance(env, str):
            self.env = gym.make(env)
        else:
            self.env = env

        self.batch_size = batch_size
        self.state_spec = utils.space_to_flat_spec(space=self.env.observation_space, name='state')
        self.action_spec = utils.space_to_flat_spec(space=self.env.action_space, name='action')
        self.set_random_seed(seed)
        self.last_value = tf.zeros((1, 1), dtype=tf.float32)

        # Data options
        self.drop_batch_reminder = drop_batch_reminder
        self.skip_count = skip_data
        self.obs_skipping = consider_obs_every
        self.shuffle_batches = shuffle_batches

        # Saving stuff:
        self.base_path = os.path.join(weights_dir, name)
        self.weights_path = dict(policy=os.path.join(self.base_path, 'policy_net'),
                                 value=os.path.join(self.base_path, 'value_net'))

        # JSON configuration file (keeps track of useful quantities)
        self.config_path = os.path.join(self.base_path, 'config.json')
        self.config = dict()

        # Statistics:
        self.statistics = utils.Statistics(mode=log_mode, name=name)

    def set_random_seed(self, seed):
        """Sets the random seed for tensorflow, numpy, python's random, and the environment"""
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            self.env.seed(seed)
            print(f'Random seed {seed} set.')

    def act(self, state):
        pass

    def update(self):
        pass

    def learn(self, *args, **kwargs):
        pass

    def evaluate(self, episodes: int, timesteps: int, render=True, seeds=None) -> list:
        rewards = []
        sample_seed = False

        if isinstance(seeds, int):
            self.set_random_seed(seed=seeds)
        elif isinstance(seeds, list):
            sample_seed = True

        for episode in range(1, episodes + 1):
            if sample_seed:
                self.set_random_seed(seed=random.choice(seeds))

            state = self.env.reset()
            episode_reward = 0.0

            for t in range(1, timesteps + 1):
                if render:
                    self.env.render()

                action = self.act(state)
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward

                self.log(actions=action, rewards=reward)

                if done or (t == timesteps):
                    print(f'Episode {episode} terminated after {t} timesteps with reward {episode_reward}.')
                    rewards.append(episode_reward)
                    break

            self.log(evaluation_reward=episode_reward)
            self.write_summaries()

        self.env.close()

        print(f'Mean reward: {round(np.mean(rewards), 2)}, std: {round(np.std(rewards), 2)}')
        return rewards

    def preprocess(self):
        @tf.function
        def preprocess_fn(*args):
            return args

        return preprocess_fn

    def pretrain(self):
        # TODO: use ImitationWrapper here?
        pass

    def log(self, **kwargs):
        self.statistics.log(**kwargs)

    def write_summaries(self):
        self.statistics.write_summaries()

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
