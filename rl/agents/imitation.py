"""Imitation Learning Agents"""

import os
import time
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import losses

from rl import utils
from rl.agents.agents import Agent

from tensorflow.keras import optimizers


# TODO: move to 'imitation' package?
# TODO: works only for PPOAgent... make general
class ImitationWrapper:
    """Imitation Learning wrapper for Agents"""

    def __init__(self, agent: Agent, policy_lr=3e-4, value_lr=3e-4, traces_dir='traces',
                 weights_dir='weights_imitation', log_mode='summary', name='imitation',
                 drop_batch_reminder=False, skip_data=0, consider_obs_every=1):
        self.agent = agent
        self.batch_size = agent.batch_size
        self.save_path = dict(policy=os.path.join(weights_dir, name, 'policy_net'),
                              value=os.path.join(weights_dir, name, 'value_net'))
        # Traces
        self.traces_dir = os.path.join(traces_dir, name)
        self.traces_names = utils.file_names(self.traces_dir, sort=True)

        # Data options
        self.drop_batch_reminder = drop_batch_reminder
        self.skip_count = skip_data
        self.obs_skipping = consider_obs_every

        # Assign the policy and value network
        self.policy_network = self.agent.policy_network
        self.value_network = self.agent.value_network

        # Define optimizer and loss
        self.policy_optimizer = optimizers.Adam(learning_rate=policy_lr)
        self.value_optimizer = optimizers.Adam(learning_rate=value_lr)

        # Statistics
        self.statistics = utils.Summary(mode=log_mode, name=name)

    def imitate(self, discount=0.99, shuffle_traces=False, shuffle_batches=False,
                repetitions=1, save_every=1, save_distinct=False, seed=None):
        print('== IMITATION LEARNING ==')
        self.agent.set_random_seed(seed)

        k = 1
        for r in range(1, repetitions + 1):
            print('\trepetition:', r)

            for i, trace in enumerate(self.load_traces(shuffle=shuffle_traces)):
                t0 = time.time()
                states, actions, rewards, done = self.interpret(trace)
                returns = utils.rewards_to_go(rewards, discount, normalize=True)

                # Train policy and value networks:
                self.train_policy(states, actions, shuffle=shuffle_batches)
                self.train_value(states, returns, shuffle=shuffle_batches)

                self.log(actions=actions, done=done, rewards=rewards, returns=returns)
                self.write_summaries()

                if k % save_every == 0:
                    self.save(info=(r, i) if save_distinct is True else None)
                k += 1
                print(f'\t\ttrace: {i} completed in time {round(time.time() - t0, 3)}s')

    def preprocess(self):
        @tf.function
        def preprocess_fn(element):
            return element

        return preprocess_fn

    def train_policy(self, states, actions, shuffle: bool):
        """One training step for policy network"""
        dataset = utils.data_to_batches((states, actions), batch_size=self.batch_size,
                                        shuffle_batches=shuffle, map_fn=self.preprocess(),
                                        drop_remainder=self.drop_batch_reminder, skip=self.skip_count,
                                        num_shards=self.obs_skipping)
        for batch in dataset:
            states_batch, true_actions = batch

            with tf.GradientTape() as tape:
                pred_actions = self.policy_network(states_batch, training=True)
                action_loss = losses.mean_absolute_error(y_true=true_actions, y_pred=pred_actions)
                action_loss = tf.reduce_mean(action_loss)

            grads = tape.gradient(action_loss, self.policy_network.trainable_variables)
            self.policy_optimizer.apply_gradients(zip(grads, self.policy_network.trainable_variables))

            self.log(loss_action=action_loss.numpy(),
                     actions_true=true_actions,
                     actions_pred=pred_actions,
                     gradients_norm_action=[tf.norm(gradient) for gradient in grads])

    def train_value(self, states, returns, shuffle: bool):
        """One training step for value network"""
        dataset = utils.data_to_batches((states, returns), batch_size=self.batch_size,
                                        shuffle_batches=shuffle, map_fn=self.preprocess(),
                                        drop_remainder=self.drop_batch_reminder, skip=self.skip_count,
                                        num_shards=self.obs_skipping)
        for batch in dataset:
            states_batch, true_values = batch

            with tf.GradientTape() as tape:
                pred_values = self.value_network(states_batch, training=True)
                value_loss = losses.mean_squared_error(y_true=true_values, y_pred=pred_values)
                value_loss = tf.reduce_mean(value_loss)

            grads = tape.gradient(value_loss, self.value_network.trainable_variables)
            self.value_optimizer.apply_gradients(zip(grads, self.value_network.trainable_variables))

            self.log(loss_value=value_loss.numpy(),
                     values_true=true_values,
                     values_pred=pred_values,
                     gradients_norm_value=[tf.norm(gradient) for gradient in grads])

    def load_traces(self, shuffle: bool):
        """Reads and generates one trace at a time"""
        if shuffle:
            traces_names = self.traces_names.copy()
            random.shuffle(traces_names)
        else:
            traces_names = self.traces_names

        for trace_name in traces_names:
            yield np.load(file=os.path.join(self.traces_dir, trace_name))

    @staticmethod
    def interpret(trace: dict) -> tuple:
        trace_keys = trace.keys()
        trace = {k: trace[k] for k in trace_keys}  # copy

        for name in ['state', 'action']:
            # check if state/action space is simple (array, i.e sum == 1) or complex (dict of arrays)
            if sum(k.startswith(name) for k in trace_keys) == 1:
                continue

            # select keys of the form 'state_xxx', then build a dict(state_x=trace['state_x'])
            keys = filter(lambda k: k.startswith(name + '_'), trace_keys)
            trace[name] = {k: trace[k] for k in keys}

        return trace['state'], trace['action'], trace['reward'], trace['done']

    def save(self, info: tuple = None):
        print('saving...')

        if isinstance(info, tuple):
            repetition, iteration = info
            self.policy_network.save(f"{self.save_path['policy']}-{repetition}-{iteration}", include_optimizer=False)
            self.value_network.save(f"{self.save_path['value']}-{repetition}-{iteration}", include_optimizer=False)
        else:
            self.policy_network.save(self.save_path['policy'], include_optimizer=False)
            self.value_network.save(self.save_path['value'], include_optimizer=False)

    def log(self, **kwargs):
        self.statistics.log(**kwargs)

    def write_summaries(self):
        self.statistics.write_summaries()
