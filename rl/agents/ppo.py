import os
import gym
import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from typing import Union

from rl import utils
from rl.agents.agents import Agent
from rl.parameters import DynamicParameter, ConstantParameter, schedules
from rl.networks.networks import PPONetwork

from tensorflow.keras import losses
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class PPOAgent(Agent):
    # TODO: implement 'action repetition'?
    # TODO: 'value_loss' a parameter that selects the loss (either 'mse' or 'huber') for the value network
    # TODO: 'noise' is broken...
    def __init__(self, *args, policy_lr: Union[float, LearningRateSchedule] = 1e-3, gamma=0.99, lambda_=0.95,
                 value_lr: Union[float, LearningRateSchedule] = 3e-4, optimization_steps=(10, 10), target_kl=False,
                 noise: Union[float, DynamicParameter] = 0.0, clip_ratio: Union[float, DynamicParameter] = 0.2,
                 load=False, name='ppo-agent', entropy_regularization: Union[float, DynamicParameter] = 0.0,
                 mixture_components=1, clip_norm=(1.0, 1.0), optimizer='adam', traces_dir: str = None,
                 network: Union[dict, PPONetwork] = None, **kwargs):
        super().__init__(*args, name=name, **kwargs)

        self.memory = None
        self.gamma = gamma
        self.lambda_ = lambda_
        self.mixture_components = mixture_components
        self.min_float = tf.constant(np.finfo(np.float32).eps, dtype=tf.float32)

        # Record:
        if isinstance(traces_dir, str):
            self.should_record = True
            self.traces_dir = utils.makedir(traces_dir, name)
        else:
            self.should_record = False

        # Entropy regularization
        if isinstance(entropy_regularization, DynamicParameter):
            self.entropy_strength = entropy_regularization
        else:
            self.entropy_strength = ConstantParameter(entropy_regularization)

        # Ratio clipping
        if isinstance(clip_ratio, float):
            assert clip_ratio >= 0.0
            self.clip_ratio = ConstantParameter(value=clip_ratio)
        else:
            assert isinstance(clip_ratio, DynamicParameter)
            self.clip_ratio = clip_ratio

        # KL
        if isinstance(target_kl, float):
            self.early_stop = True
            self.target_kl = tf.constant(target_kl * 1.5)
        else:
            self.early_stop = False

        # TODO: handle complex action spaces (make use of Agent.action_spec)
        # Action space
        action_space = self.env.action_space

        if isinstance(action_space, gym.spaces.Box):
            self.num_actions = action_space.shape[0]

            # continuous:
            if action_space.is_bounded():
                self.distribution_type = 'beta'

                self.action_low = tf.constant(action_space.low, dtype=tf.float32)
                self.action_high = tf.constant(action_space.high, dtype=tf.float32)
                self.action_range = tf.constant(action_space.high - action_space.low,
                                                dtype=tf.float32)

                self.convert_action = lambda a: (a * self.action_range + self.action_low)[0].numpy()
            else:
                self.distribution_type = 'gaussian'
                self.convert_action = lambda a: a[0].numpy()
        else:
            # discrete:
            self.distribution_type = 'categorical'

            if isinstance(action_space, gym.spaces.MultiDiscrete):
                # make sure all discrete components of the space have the same number of classes
                assert np.all(action_space.nvec == action_space.nvec[0])

                self.num_actions = action_space.nvec.shape[0]
                self.num_classes = action_space.nvec[0] + 1  # to include the last class, i.e. 0 to K (not 0 to k-1)
                self.convert_action = lambda a: tf.cast(a[0], dtype=tf.int32).numpy()
            else:
                self.num_actions = 1
                self.num_classes = action_space.n
                self.convert_action = lambda a: tf.cast(tf.squeeze(a), dtype=tf.int32).numpy()

        # Gaussian noise (for exploration)
        if isinstance(noise, float):
            self.noise_std = ConstantParameter(value=noise)
        elif isinstance(noise, DynamicParameter):
            self.noise_std = noise
        else:
            raise ValueError("Noise should be an instance of float or DynamicParameter!")

        # print('state_shape:', self.state_shape)
        print('state_spec:', self.state_spec)
        print('action_shape:', self.num_actions)
        print('distribution:', self.distribution_type)

        # Gradient clipping:
        if clip_norm is None:
            self.should_clip_policy_grads = False
            self.should_clip_value_grads = False
        else:
            assert isinstance(clip_norm, tuple)

            if clip_norm[0] is None:
                self.should_clip_policy_grads = False
            else:
                self.should_clip_policy_grads = True
                self.grad_norm_policy = tf.constant(clip_norm[0], dtype=tf.float32)

            if clip_norm[1] is None:
                self.should_clip_value_grads = False
            else:
                self.should_clip_value_grads = True
                self.grad_norm_value = tf.constant(clip_norm[1], dtype=tf.float32)

        # Optimization
        self.policy_optimizer = utils.get_optimizer_by_name(optimizer, learning_rate=policy_lr)
        self.value_optimizer = utils.get_optimizer_by_name(optimizer, learning_rate=value_lr)
        self.optimization_steps = dict(policy=optimization_steps[0], value=optimization_steps[1])

        self.has_schedule_policy = isinstance(policy_lr, schedules.Schedule)
        self.has_schedule_value = isinstance(value_lr, schedules.Schedule)
        self.policy_lr = policy_lr
        self.value_lr = value_lr

        # Incremental mean and std of returns and advantages (used to normalize them)
        self.returns = utils.IncrementalStatistics()
        self.advantages = utils.IncrementalStatistics()

        # Networks & Loading
        if isinstance(network, PPONetwork):
            self.network = network(agent=self)

        elif isinstance(network, dict):
            self.network = network.pop('network', PPONetwork)(agent=self, **network)
        else:
            self.network = PPONetwork(agent=self)

        if load:
            self.load()

    def act(self, state):
        action = self.network.act(inputs=utils.to_tensor(state))
        return self.convert_action(action)

    def predict(self, state):
        return self.network.predict(inputs=state)

    def update(self):
        t0 = time.time()

        # Compute advantages and returns:
        advantages = self.get_advantages()
        returns = self.get_returns()

        self.log(returns=returns, advantages=advantages, values=self.memory.values)

        # Prepare data: (states, returns) and (states, advantages, actions, log_prob)
        value_batches = self.get_value_batches(returns)
        policy_batches = self.get_policy_batches(advantages)

        # Policy network optimization:
        for opt_step in range(self.optimization_steps['policy']):
            for data_batch in policy_batches:
                total_loss, kl, policy_grads = self.network.update_step_policy(data_batch)

                self.log(loss_total=total_loss.numpy(),
                         lr_policy=self.policy_lr.lr if self.has_schedule_policy else self.policy_lr,
                         gradients_norm_policy=[tf.norm(gradient) for gradient in policy_grads])

                if self.distribution_type == 'categorical':
                    logits = self.network.policy.get_layer(name='logits')
                    weights, bias = logits.trainable_variables

                    self.log(weights_logits=tf.norm(weights), bias_logits=tf.norm(weights))

                elif self.distribution_type == 'beta':
                    alpha = self.network.policy.get_layer(name='alpha')
                    beta = self.network.policy.get_layer(name='beta')

                    weights_a, bias_a = alpha.trainable_variables
                    weights_b, bias_b = beta.trainable_variables

                    self.log(weights_alpha=tf.norm(weights_a), bias_alpha=tf.norm(bias_a),
                             weights_beta=tf.norm(weights_b), bias_beta=tf.norm(bias_b))

            # Stop early if target_kl is reached:
            if self.early_stop and (kl > self.target_kl):
                self.log(early_stop=opt_step)
                print(f'early stop at step {opt_step}.')
                break

        # Value network optimization:
        for _ in range(self.optimization_steps['value']):
            for data_batch in value_batches:
                value_loss, value_grads = self.network.update_step_value(data_batch)

                self.log(loss_value=value_loss.numpy(),
                         lr_value=self.value_lr.lr if self.has_schedule_value else self.value_lr,
                         gradients_norm_value=[tf.norm(gradient) for gradient in value_grads])

        print(f'Update took {round(time.time() - t0, 3)}s')

    def get_value_batches(self, returns, **kwargs):
        """Computes batches of data for updating the value network"""
        return utils.data_to_batches(tensors=(self.memory.states, returns), batch_size=self.batch_size,
                                     drop_remainder=self.drop_batch_reminder, skip=self.skip_count,
                                     map_fn=self.preprocess(),
                                     num_shards=self.obs_skipping, shuffle_batches=self.shuffle_batches)

    def get_policy_batches(self, advantages, **kwargs):
        """Computes batches of data for updating the policy network"""
        return utils.data_to_batches(tensors=(self.memory.states, advantages, self.memory.actions,
                                              self.memory.log_probabilities),
                                     batch_size=self.batch_size, drop_remainder=self.drop_batch_reminder,
                                     skip=self.skip_count, num_shards=self.obs_skipping,
                                     shuffle_batches=self.shuffle_batches, map_fn=self.preprocess())

    # @tf.function
    def value_objective(self, batch):
        states, returns = batch
        values = self.network.value(states, training=True)
        # tf.print(values)
        # tf.print(returns)
        return tf.reduce_mean(losses.mean_squared_error(y_true=returns, y_pred=values))

    def policy_objective(self, batch):
        """PPO-Clip Objective"""
        states, advantages, actions, old_log_probabilities = batch
        new_policy: tfp.distributions.Distribution = self.network.policy(states, training=True)

        if self.distribution_type == 'categorical' and self.num_actions == 1:
            batch_size = tf.shape(actions)[0]
            actions = tf.reshape(actions, shape=batch_size)

            # new_log_prob = new_policy.log_prob(tf.reshape(actions, shape=batch_size))
            new_log_prob = new_policy.log_prob(actions)
            new_log_prob = tf.reshape(new_log_prob, shape=(batch_size, self.num_actions))
        else:
            # round samples (actions) before computing density:
            # motivation: https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Beta
            actions = tf.clip_by_value(actions, self.min_float, 1.0 - self.min_float)
            new_log_prob = new_policy.log_prob(actions)

        kl_divergence = utils.kl_divergence(old_log_probabilities, new_log_prob)

        # Entropy
        entropy = new_policy.entropy()
        entropy_coeff = self.entropy_strength()
        entropy_penalty = -entropy_coeff * tf.reduce_mean(entropy)

        # Compute the probability ratio between the current and old policy
        ratio = tf.math.exp(new_log_prob - old_log_probabilities)

        # Compute the clipped ratio times advantage
        clip_value = self.clip_ratio()
        clipped_ratio = tf.clip_by_value(ratio, clip_value_min=1.0 - clip_value, clip_value_max=1.0 + clip_value)

        # Loss = min { ratio * A, clipped_ratio * A } + entropy_term
        policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
        total_loss = policy_loss + entropy_penalty

        # Log stuff
        self.log(ratio=ratio, log_prob=new_log_prob, entropy=entropy, entropy_coeff=entropy_coeff,
                 ratio_clip=clip_value, kl_divergence=kl_divergence, loss_policy=policy_loss.numpy(),
                 loss_entropy=entropy_penalty.numpy())

        return total_loss, tf.reduce_mean(kl_divergence)

    def get_advantages(self):
        advantages = utils.gae(rewards=self.memory.rewards, values=self.memory.values, gamma=self.gamma,
                               lambda_=self.lambda_, normalize=False)
        self.advantages.update(advantages)
        self.log(advantages_mean=[self.advantages.mean], advantages_std=[self.advantages.std])

        return tf.cast((advantages - self.advantages.mean) / self.advantages.std, dtype=tf.float32)

    def get_returns(self):
        returns = utils.rewards_to_go(rewards=self.memory.rewards, discount=self.gamma, normalize=False)
        self.returns.update(returns)
        self.log(returns_mean=[self.returns.mean], returns_std=[self.returns.std])

        # normalize using running mean and std:
        return (returns - self.returns.mean) / self.returns.std

    def learn(self, episodes: int, timesteps: int, save_every: Union[bool, str, int] = False,
              render_every: Union[bool, str, int] = False):
        if save_every is False:
            save_every = episodes + 1
        elif save_every is True:
            save_every = 1
        elif save_every == 'end':
            save_every = episodes

        if render_every is False:
            render_every = episodes + 1
        elif render_every is True:
            render_every = 1

        try:
            for episode in range(1, episodes + 1):
                self.reset()
                self.memory = PPOMemory(state_spec=self.state_spec, num_actions=self.num_actions)

                state = self.env.reset()
                state = utils.to_tensor(state)

                # TODO: temporary fix (should be buggy as well...)
                if isinstance(state, dict):
                    state = {f'state_{k}': v for k, v in state.items()}

                episode_reward = 0.0
                t0 = time.time()
                render = episode % render_every == 0

                for t in range(1, timesteps + 1):
                    if render:
                        self.env.render()

                    # Compute action, log_prob, and value
                    action, mean, std, log_prob, value = self.predict(state)
                    action_env = self.convert_action(action)

                    # Environment step
                    next_state, reward, done, _ = self.env.step(action_env)
                    episode_reward += reward

                    self.log(actions=action, action_env=action_env, rewards=reward,
                             distribution_mean=mean, distribution_std=std)

                    self.memory.append(state, action, reward, value, log_prob)
                    state = utils.to_tensor(next_state)

                    if isinstance(state, dict):
                        state = {f'state_{k}': v for k, v in state.items()}

                    # check whether a termination (terminal state or end of a transition) is reached:
                    if done or (t == timesteps):
                        print(f'Episode {episode} terminated after {t} timesteps in {round((time.time() - t0), 3)}s ' +
                              f'with reward {round(episode_reward, 3)}.')
                        self.memory.end_trajectory(last_value=self.last_value if done else self.network.value(state))
                        break

                self.update()
                self.log(episode_rewards=episode_reward)
                self.write_summaries()

                if self.should_record:
                    self.memory.serialize(episode, save_path=self.traces_dir)

                if episode % save_every == 0:
                    self.save()
        finally:
            print('closing...')
            self.env.close()

    def summary(self):
        self.network.summary()

    def save_weights(self):
        print('saving weights...')
        self.network.save_weights()

    def load_weights(self):
        print('loading weights...')
        self.network.load_weights()

    def save_config(self):
        print('save config')
        self.update_config(returns=self.returns.as_dict(), advantages=self.advantages.as_dict())
        super().save_config()

    def load_config(self):
        print('load config')
        super().load_config()
        self.returns.set(**self.config['returns'])
        self.advantages.set(**self.config['advantages'])

    def reset(self):
        super().reset()
        self.network.reset()


class PPOMemory:
    """Recent memory used in PPOAgent"""

    # TODO: define what to store from a specification (dict: str -> (shape, dtype))
    def __init__(self, state_spec: dict, num_actions: int):
        if list(state_spec.keys()) == ['state']:
            # Simple state-space
            self.states = tf.zeros(shape=(0,) + state_spec.get('state'), dtype=tf.float32)
            self.simple_state = True
        else:
            # Complex state-space
            self.states = dict()
            self.simple_state = False

            for name, shape in state_spec.items():
                self.states[name] = tf.zeros(shape=(0,) + shape, dtype=tf.float32)

        self.rewards = tf.zeros(shape=(0,), dtype=tf.float32)
        self.values = tf.zeros(shape=(0, 1), dtype=tf.float32)
        self.actions = tf.zeros(shape=(0, num_actions), dtype=tf.float32)
        self.log_probabilities = tf.zeros(shape=(0, num_actions), dtype=tf.float32)

    # TODO: use kwargs to define what to append and where to store
    def append(self, state, action, reward, value, log_prob):
        if self.simple_state:
            self.states = tf.concat([self.states, state], axis=0)
        else:
            assert isinstance(state, dict)

            for k, v in state.items():
                self.states[k] = tf.concat([self.states[k], v], axis=0)

        self.actions = tf.concat([self.actions, tf.cast(action, dtype=tf.float32)], axis=0)
        self.rewards = tf.concat([self.rewards, [reward]], axis=0)
        self.values = tf.concat([self.values, value], axis=0)
        self.log_probabilities = tf.concat([self.log_probabilities, log_prob], axis=0)

    def end_trajectory(self, last_value):
        """Terminates the current trajectory by adding the value of the terminal state"""
        self.rewards = tf.concat([self.rewards, last_value[0]], axis=0)
        self.values = tf.concat([self.values, last_value], axis=0)

    def serialize(self, episode: int, save_path: str):
        """Writes to file (npz - numpy compressed format) all the transitions collected so far"""
        # Trace's file path:
        filename = f'trace-{episode}-{time.strftime("%Y%m%d-%H%M%S")}.npz'
        trace_path = os.path.join(save_path, filename)

        # Select data to save
        buffer = dict(reward=self.rewards, action=self.actions)

        if self.simple_state:
            buffer['state'] = self.states
        else:
            for key, value in self.states.items():
                buffer[key] = value

        # Save buffer
        np.savez_compressed(file=trace_path, **buffer)
        print(f'Traces "{filename}" saved.')
