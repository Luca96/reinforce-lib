import os
import gym
import random
import numpy as np
import tensorflow as tf

from rl import utils
from typing import List, Dict
from tensorflow.keras import layers


# TODO: implement RandomAgent
# TODO: actor-critic agent interface (to include policy/value network as well loading/saving)?
# TODO: imitation learning agents, e.g. CIL agent (conditional imitation learning)

class Agent:
    """Agent abstract class"""
    # TODO: check random seed issue
    def __init__(self, env: gym.Env, seed=None, weights_dir='weights', name='agent', log_mode='summary'):
        self.env = env
        self.state_spec = utils.space_to_flat_spec(space=self.env.observation_space, name='state')
        self.action_spec = utils.space_to_flat_spec(space=self.env.action_space, name='action')

        # Set random seed:
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            self.env.seed(seed)
            print(f'Random seed {seed} set.')

        # Saving stuff:
        self.base_path = os.path.join(weights_dir, name)
        self.save_path = dict(policy=os.path.join(self.base_path, 'policy_net'),
                              value=os.path.join(self.base_path, 'value_net'))
        # Statistics:
        self.statistics = utils.Statistics(mode=log_mode, name=name)

    def act(self, state):
        pass

    def update(self, batch_size: int):
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
            layer = layers.Input(shape=shape, dtype=tf.float32, name=name)
            input_layers[name] = layer

        return input_layers
