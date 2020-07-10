import os
import gym
import random
import numpy as np
import tensorflow as tf

from rl import utils
from typing import List, Dict
from tensorflow.keras import layers


# TODO: implement RandomAgent
# TODO: actor-critic agent interface (to include policy/value network as well as loading/saving)?
class Agent:
    """Agent abstract class"""
    def __init__(self, env: gym.Env, batch_size: int, seed=None, weights_dir='weights', name='agent',
                 log_mode='summary', drop_batch_reminder=False, skip_data=0, consider_obs_every=1,
                 shuffle_batches=False):
        self.env = env
        self.batch_size = batch_size
        self.state_spec = utils.space_to_flat_spec(space=self.env.observation_space, name='state')
        self.action_spec = utils.space_to_flat_spec(space=self.env.action_space, name='action')
        self.set_random_seed(seed)

        # Data options
        self.drop_batch_reminder = drop_batch_reminder
        self.skip_count = skip_data
        self.obs_skipping = consider_obs_every
        self.shuffle_batches = shuffle_batches

        # Saving stuff:
        self.base_path = os.path.join(weights_dir, name)
        self.save_path = dict(policy=os.path.join(self.base_path, 'policy_net'),
                              value=os.path.join(self.base_path, 'value_net'))
        # Statistics:
        self.statistics = utils.Statistics(mode=log_mode, name=name)

    def set_random_seed(self, seed):
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

    def evaluate(self, episodes: int, timesteps: int, render=True) -> list:
        rewards = []

        for episode in range(1, episodes + 1):
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

    def pretrain(self):
        pass

    def log(self, **kwargs):
        self.statistics.log(**kwargs)

    def write_summaries(self):
        self.statistics.write_summaries()

    def summary(self):
        """Networks summary"""
        raise NotImplementedError

    def _get_input_layers(self) -> Dict[str, layers.Input]:
        """Handles arbitrary complex state-spaces"""
        input_layers = dict()

        for name, shape in self.state_spec.items():
            if self.drop_batch_reminder:
                layer = layers.Input(shape=shape, batch_size=self.batch_size, dtype=tf.float32, name=name)
            else:
                layer = layers.Input(shape=shape, dtype=tf.float32, name=name)

            input_layers[name] = layer

        return input_layers