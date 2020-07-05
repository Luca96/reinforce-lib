import os
import gym
import math
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime


# TODO: implement RandomAgent
# TODO: actor-critic agent interface (to include policy/value network as well loading/saving)?
# TODO: imitation learning agents, e.g. CIL agent (conditional imitation learning)

class Agent:
    """Agent abstract class"""
    # TODO: check random seed issue
    def __init__(self, env: gym.Env, seed=None, weights_dir='weights', name='agent', use_log=False, use_summary=False):
        self.env = env

        # Logging
        self.use_log = use_log
        self.use_summary = use_summary

        if self.use_summary:
            self.summary_dir = os.path.join('logs', name, datetime.now().strftime("%Y%m%d-%H%M%S"))
            self.tf_summary_writer = tf.summary.create_file_writer(self.summary_dir, max_queue=5)

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

        # TODO: made a 'statistics class'
        # Statistics: (value_list, step_num)
        self.stats = dict()

    def act(self, state):
        pass

    def update(self, batch_size: int):
        pass

    def learn(self, *args, **kwargs):
        pass

    def evaluate(self):
        pass

    def pretrain(self):
        pass

    def log(self, **kwargs):
        if self.use_log:
            for key, value in kwargs.items():
                if hasattr(value, '__iter__'):
                    self.stats[key][0].extend(value)
                else:
                    self.stats[key][0].append(value)

    def write_summaries(self):
        with self.tf_summary_writer.as_default():
            for key, (values, step) in self.stats.items():

                for i, value in enumerate(values):
                    tf.summary.scalar(name=key, data=np.squeeze(value), step=step + i)

                # clear value_list, update step
                self.stats[key][1] += len(values)
                self.stats[key][0].clear()

    def plot_statistics(self, colormap='Set3'):  # Pastel1, Set3, tab20b, tab20c
        """Colormaps: https://matplotlib.org/tutorials/colors/colormaps.html"""
        num_plots = len(self.stats.keys())
        cmap = plt.get_cmap(name=colormap)
        rows = round(math.sqrt(num_plots))
        cols = math.ceil(math.sqrt(num_plots))

        for k, (key, value) in enumerate(self.stats.items()):
            plt.subplot(rows, cols, k + 1)
            plt.plot(value, color=cmap(k + 1))
            plt.title(key)

        plt.show()
