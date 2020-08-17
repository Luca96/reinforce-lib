import os
import gym
import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from typing import Union

from rl import utils
from rl.agents.agents import Agent
from rl.parameters import DynamicParameter, ConstantParameter, schedules, ParameterWrapper
from rl.networks.networks import PPONetwork

from tensorflow.keras import losses
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class PPOAgent(Agent):
    # TODO: implement 'action repetition'?
    # TODO: 'value_loss' a parameter that selects the loss (either 'mse' or 'huber') for the value network
    # TODO: implement value function clipping?
    # TODO: dynamic-parameters: kl, noise, clip_norm...
    # TODO: polyak factor to 0.99 or 0.95
    def __init__(self, *args, policy_lr: Union[float, LearningRateSchedule] = 1e-3, gamma=0.99, lambda_=0.95,
                 value_lr: Union[float, LearningRateSchedule] = 3e-4, optimization_steps=(1, 1), target_kl=False,
                 clip_ratio: Union[float, LearningRateSchedule, DynamicParameter] = 0.2, load=False, name='ppo-agent',
                 entropy_regularization: Union[float, LearningRateSchedule, DynamicParameter] = 0.0, optimizer='adam',
                 clip_norm=(1.0, 1.0), mixture_components=1, network: Union[dict, PPONetwork] = None,
                 accumulate_gradients_over_batches=False, update_frequency=1, polyak=1.0, **kwargs):
        assert 0.0 < polyak <= 1.0

        super().__init__(*args, name=name, **kwargs)

        self.memory: PPOMemory = None
        self.gamma = gamma
        self.lambda_ = lambda_
        self.mixture_components = mixture_components

        # Entropy regularization
        if isinstance(entropy_regularization, float):
            self.entropy_strength = ConstantParameter(value=entropy_regularization)

        elif isinstance(entropy_regularization, LearningRateSchedule):
            self.entropy_strength = ParameterWrapper(entropy_regularization)

        # Ratio clipping
        if isinstance(clip_ratio, float):
            assert clip_ratio >= 0.0
            self.clip_ratio = ConstantParameter(value=clip_ratio)

        elif isinstance(clip_ratio, LearningRateSchedule):
            self.clip_ratio = ParameterWrapper(clip_ratio)

        # KL
        if isinstance(target_kl, float):
            self.early_stop = True
            self.target_kl = tf.constant(target_kl * 1.5)
        else:
            self.early_stop = False

        # Action space
        self._init_action_space()

        print('state_spec:', self.state_spec)
        print('action_shape:', self.num_actions)
        print('distribution:', self.distribution_type)

        # Gradient clipping:
        self._init_gradient_clipping(clip_norm)

        # Networks & Loading
        self.weights_path = dict(policy=os.path.join(self.base_path, 'policy_net'),
                                 value=os.path.join(self.base_path, 'value_net'))

        if isinstance(network, dict):
            network_class = network.pop('network', PPONetwork)

            if network_class is PPONetwork:
                # policy/value-specific arguments
                policy_args = network.pop('policy', {})
                value_args = network.pop('value', policy_args)

                # common arguments
                for k, v in network.items():
                    if k not in policy_args:
                        policy_args[k] = v

                    if k not in value_args:
                        value_args[k] = v

                self.network = network_class(agent=self, policy=policy_args, value=value_args, **network)
            else:
                self.network = network_class(agent=self, **network)
        else:
            self.network = PPONetwork(agent=self, policy={}, value={})

        if load:
            self.load()

        # Optimization
        self.accumulate_gradients_over_batches = accumulate_gradients_over_batches
        self.update_frequency = update_frequency
        self.episode_count = 0
        self.episodic_policy_gradients = None
        self.episodic_value_gradients = None

        if self.update_frequency > 1:
            self.accumulate_gradients_over_batches = True

        self.policy_lr = self._init_lr_schedule(policy_lr, config=self.config.get('policy_lr', {}))
        self.value_lr = self._init_lr_schedule(value_lr, config=self.config.get('value_lr', {}))

        self.policy_optimizer = utils.get_optimizer_by_name(optimizer, learning_rate=self.policy_lr)
        self.value_optimizer = utils.get_optimizer_by_name(optimizer, learning_rate=self.value_lr)
        self.optimization_steps = dict(policy=optimization_steps[0], value=optimization_steps[1])

        self.should_polyak_average = polyak < 1.0
        self.polyak_coeff = polyak

    @staticmethod
    def _init_lr_schedule(lr: Union[float, LearningRateSchedule], config: dict):
        if isinstance(lr, float):
            return schedules.ConstantSchedule(lr)

        elif not isinstance(lr, schedules.Schedule) and isinstance(lr, LearningRateSchedule):
            return schedules.ScheduleWrapper(lr_schedule=lr, offset=config.get('step_offset', 0))

        return lr

    def _init_gradient_clipping(self, clip_norm: Union[tuple, float, None]):
        if clip_norm is None:
            self.should_clip_policy_grads = False
            self.should_clip_value_grads = False

        elif isinstance(clip_norm, float):
            assert clip_norm > 0.0
            self.should_clip_policy_grads = True
            self.should_clip_value_grads = True

            self.grad_norm_policy = clip_norm
            self.grad_norm_value = clip_norm
        else:
            assert isinstance(clip_norm, tuple)

            if clip_norm[0] is None:
                self.should_clip_policy_grads = False
            else:
                assert isinstance(clip_norm[0], float)
                assert clip_norm[0] > 0.0

                self.should_clip_policy_grads = True
                self.grad_norm_policy = tf.constant(clip_norm[0], dtype=tf.float32)

            if clip_norm[1] is None:
                self.should_clip_value_grads = False
            else:
                assert isinstance(clip_norm[1], float)
                assert clip_norm[1] > 0.0

                self.should_clip_value_grads = True
                self.grad_norm_value = tf.constant(clip_norm[1], dtype=tf.float32)

    # TODO: handle complex action spaces (make use of Agent.action_spec)
    def _init_action_space(self):
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

    def act(self, state):
        action = self.network.act(inputs=utils.to_tensor(state))
        return self.convert_action(action)

    def predict(self, state):
        return self.network.predict(inputs=state)

    def update(self):
        t0 = time.time()
        self.episode_count += 1

        # Compute advantages and returns:
        denorm_returns = self.memory.compute_returns(discount=self.gamma)
        denorm_values = self.memory.compute_advantages(self.gamma, self.lambda_)

        self.log(returns=self.memory.returns, advantages=self.memory.advantages, values=self.memory.values,
                 returns_denorm=denorm_returns, values_denorm=denorm_values)

        # Prepare data:
        value_batches = self.get_value_batches()
        policy_batches = self.get_policy_batches()

        # Policy network optimization:
        for opt_step in range(self.optimization_steps['policy']):
            gradients = None
            num_batches = 0

            for data_batch in policy_batches:
                total_loss, kl, policy_grads = self.get_policy_gradients(data_batch)

                if self.accumulate_gradients_over_batches:
                    num_batches += 1
                    gradients = utils.accumulate_gradients(policy_grads, gradients)
                else:
                    self.update_policy(policy_grads)
                    self.log(lr_policy=self.policy_lr.lr)

                self.log(loss_total=total_loss.numpy(),
                         gradients_norm_policy=[tf.norm(gradient) for gradient in policy_grads])

            if self.accumulate_gradients_over_batches:
                batch_gradients = utils.average_gradients(gradients, n=num_batches)
                batch_gradients, updated = self.update_policy(batch_gradients)

                if updated:
                    self.log(gradients_norm_policy_update=[tf.norm(g) for g in batch_gradients],
                             lr_policy=self.policy_lr.lr)
                else:
                    self.log(gradients_norm_policy_batch=[tf.norm(g) for g in batch_gradients])

            # # Stop early if target_kl is reached:
            # if self.early_stop and (kl > self.target_kl):
            #     self.log(early_stop=opt_step)
            #     print(f'early stop at step {opt_step}.')
            #     break

            if self.distribution_type == 'categorical':
                logits = self.network.policy.get_layer(name='logits')

                if isinstance(logits, tf.keras.layers.Dense):
                    weights, bias = logits.trainable_variables

                    self.log(weights_logits=tf.norm(weights), bias_logits=tf.norm(weights))

            elif self.distribution_type == 'beta':
                alpha = self.network.policy.get_layer(name='alpha')
                beta = self.network.policy.get_layer(name='beta')

                if isinstance(alpha, tf.keras.layers.Dense) and isinstance(beta, tf.keras.layers.Dense):
                    weights_a, bias_a = alpha.trainable_variables
                    weights_b, bias_b = beta.trainable_variables

                    self.log(weights_alpha=tf.norm(weights_a), bias_alpha=tf.norm(bias_a),
                             weights_beta=tf.norm(weights_b), bias_beta=tf.norm(bias_b))

        # Value network optimization:
        for _ in range(self.optimization_steps['value']):
            gradients = None
            num_batches = 0

            for data_batch in value_batches:
                value_loss, value_grads = self.get_value_gradients(data_batch)

                if self.accumulate_gradients_over_batches:
                    num_batches += 1
                    gradients = utils.accumulate_gradients(value_grads, gradients)
                else:
                    self.update_value(value_grads)
                    self.log(lr_value=self.value_lr.lr)

                self.log(loss_value=value_loss.numpy(),
                         gradients_norm_value=[tf.norm(gradient) for gradient in value_grads])

            if self.accumulate_gradients_over_batches:
                batch_gradients = utils.average_gradients(gradients, n=num_batches)
                batch_gradients, updated = self.update_value(batch_gradients)

                if updated:
                    self.log(gradients_norm_value_update=[tf.norm(g) for g in batch_gradients],
                             lr_value=self.value_lr.lr)
                else:
                    self.log(gradients_norm_value_batch=[tf.norm(g) for g in batch_gradients])

        print(f'Update took {round(time.time() - t0, 3)}s')

    def get_policy_gradients(self, batch):
        with tf.GradientTape() as tape:
            loss, kl = self.policy_objective(batch)

        gradients = tape.gradient(loss, self.network.policy.trainable_variables)
        return loss, kl, gradients

    def update_policy(self, gradients) -> (list, bool):
        if self.update_frequency > 1:
            self.episodic_policy_gradients = utils.accumulate_gradients(gradients, self.episodic_policy_gradients)

            if self.episode_count % self.update_frequency == 0:
                gradients = utils.average_gradients(self.episodic_policy_gradients, n=self.update_frequency)
                self.episodic_policy_gradients = None
            else:
                # prevents the following code to update the policy network
                return gradients, False

        # update policy network
        if self.should_clip_policy_grads:
            gradients = [tf.clip_by_norm(grad, clip_norm=self.grad_norm_policy) for grad in gradients]

        if self.should_polyak_average:
            old_weights = self.network.policy.get_weights()
            self.network.update_old_policy(old_weights)

            self.policy_optimizer.apply_gradients(zip(gradients, self.network.policy.trainable_variables))
            utils.polyak_averaging(self.network.policy, old_weights, alpha=self.polyak_coeff)
        else:
            self.network.update_old_policy()
            self.policy_optimizer.apply_gradients(zip(gradients, self.network.policy.trainable_variables))

        return gradients, True

    def get_value_gradients(self, batch):
        with tf.GradientTape() as tape:
            loss = self.value_objective(batch)

        gradients = tape.gradient(loss, self.network.value.trainable_variables)
        return loss, gradients

    def update_value(self, gradients) -> (list, bool):
        if self.update_frequency > 1:
            self.episodic_value_gradients = utils.accumulate_gradients(gradients, self.episodic_value_gradients)

            if self.episode_count % self.update_frequency == 0:
                gradients = utils.average_gradients(self.episodic_value_gradients, n=self.update_frequency)
                self.episodic_value_gradients = None
            else:
                # prevents the following code to update the value network
                return gradients, False

        # update value network
        if self.should_clip_value_grads:
            gradients = [tf.clip_by_norm(grad, clip_norm=self.grad_norm_value) for grad in gradients]

        if self.should_polyak_average:
            old_weights = self.network.value.get_weights()
            self.value_optimizer.apply_gradients(zip(gradients, self.network.value.trainable_variables))
            utils.polyak_averaging(self.network.value, old_weights, alpha=self.polyak_coeff)
        else:
            self.value_optimizer.apply_gradients(zip(gradients, self.network.value.trainable_variables))

        return gradients, True

    def get_value_batches(self):
        """Computes batches of data for updating the value network"""
        return utils.data_to_batches(tensors=self.value_batch_tensors(), batch_size=self.batch_size,
                                     drop_remainder=self.drop_batch_reminder, skip=self.skip_count,
                                     # map_fn=self.preprocess(),
                                     shuffle=True, shuffle_batches=False,
                                     num_shards=self.obs_skipping)

    def value_batch_tensors(self) -> Union[tuple, dict]:
        """Defines which data to use in `get_value_batches()`"""
        return self.memory.states, self.memory.returns

    def get_policy_batches(self):
        """Computes batches of data for updating the policy network"""
        return utils.data_to_batches(tensors=self.policy_batch_tensors(), batch_size=self.batch_size,
                                     drop_remainder=self.drop_batch_reminder, skip=self.skip_count,
                                     num_shards=self.obs_skipping, shuffle=self.shuffle_data,
                                     # map_fn=self.preprocess(),
                                     shuffle_batches=self.shuffle_batches)

    def policy_batch_tensors(self) -> Union[tuple, dict]:
        """Defines which data to use in `get_policy_batches()`"""
        return self.memory.states, self.memory.advantages, self.memory.actions, self.memory.log_probabilities

    @tf.function
    def value_objective(self, batch):
        states, returns = batch[:2]
        values = self.network.value(states, training=True)

        return 0.5 * tf.reduce_mean(losses.mean_squared_error(y_true=returns, y_pred=values))

    def policy_objective(self, batch):
        """PPO-Clip Objective"""
        states, advantages, actions, old_log_probabilities = batch[:4]
        new_policy: tfp.distributions.Distribution = self.network.policy(states, training=True)

        if self.distribution_type == 'categorical' and self.num_actions == 1:
            batch_size = tf.shape(actions)[0]
            actions = tf.reshape(actions, shape=batch_size)

            new_log_prob = new_policy.log_prob(actions)
            new_log_prob = tf.reshape(new_log_prob, shape=(batch_size, self.num_actions))
        else:
            # round samples (actions) before computing density:
            # motivation: https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Beta
            actions = tf.clip_by_value(actions, utils.EPSILON, 1.0 - utils.EPSILON)
            new_log_prob = new_policy.log_prob(actions)

        kl_divergence = utils.kl_divergence(old_log_probabilities, new_log_prob)

        # Entropy
        entropy = new_policy.entropy()
        entropy_penalty = -self.entropy_strength() * tf.reduce_mean(entropy)

        # Compute the probability ratio between the current and old policy
        ratio = tf.math.exp(new_log_prob - old_log_probabilities)

        # Compute the clipped ratio times advantage
        clip_value = self.clip_ratio()
        clipped_ratio = tf.clip_by_value(ratio, clip_value_min=1.0 - clip_value, clip_value_max=1.0 + clip_value)

        # Loss = min { ratio * A, clipped_ratio * A } + entropy_term
        policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
        total_loss = policy_loss + entropy_penalty

        # Log stuff
        self.log(ratio=ratio, log_prob=new_log_prob, entropy=entropy, entropy_coeff=self.entropy_strength.value,
                 ratio_clip=clip_value, kl_divergence=kl_divergence, loss_policy=policy_loss.numpy(),
                 loss_entropy=entropy_penalty.numpy())

        return total_loss, tf.reduce_mean(kl_divergence)

    def learn(self, episodes: int, timesteps: int, save_every: Union[bool, str, int] = False,
              render_every: Union[bool, str, int] = False, close=True):
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

        preprocess_fn = self.preprocess()
        self.episode_count = 0

        try:
            for episode in range(1, episodes + 1):
                self.reset()
                self.memory = PPOMemory(state_spec=self.state_spec, num_actions=self.num_actions)

                state = self.env.reset()
                state = utils.to_tensor(state)
                state = preprocess_fn(state)

                # TODO: temporary fix (shouldn't work for deeper nesting...)
                if isinstance(state, dict):
                    state = {f'state_{k}': v for k, v in state.items()}

                episode_reward = 0.0
                t0 = time.time()
                render = episode % render_every == 0

                for t in range(1, timesteps + 1):
                    if render:
                        self.env.render()

                    # Agent prediction
                    action, mean, std, log_prob, value = self.predict(state)
                    action_env = self.convert_action(action)

                    # Environment step
                    next_state, reward, done, _ = self.env.step(action_env)
                    episode_reward += reward

                    self.log(actions=action, action_env=action_env, rewards=reward,
                             distribution_mean=mean, distribution_std=std)

                    self.memory.append(state, action, reward, value, log_prob)
                    state = utils.to_tensor(next_state)
                    state = preprocess_fn(state)

                    if isinstance(state, dict):
                        state = {f'state_{k}': v for k, v in state.items()}

                    # check whether a termination (terminal state or end of a transition) is reached:
                    if done or (t == timesteps):
                        print(f'Episode {episode} terminated after {t} timesteps in {round((time.time() - t0), 3)}s ' +
                              f'with reward {round(episode_reward, 3)}.')

                        last_value = self.last_value if done else self.network.predict_last_value(state)
                        self.memory.end_trajectory(last_value)
                        break

                self.update()
                self.log(episode_rewards=episode_reward)
                self.write_summaries()

                if self.should_record:
                    self.record(episode)

                if episode % save_every == 0:
                    self.save()
        finally:
            if close:
                print('closing...')
                self.env.close()

    def record(self, episode: int):
        self.memory.serialize(episode, save_path=self.traces_dir)

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
        self.update_config(policy_lr=self.policy_lr.serialize(), value_lr=self.value_lr.serialize(),
                           entropy_strength=self.entropy_strength.serialize(), clip_ratio=self.clip_ratio.serialize())
        super().save_config()

    def load_config(self):
        print('load config')
        super().load_config()

        self.entropy_strength.load(config=self.config.get('entropy_strength', 0))
        self.clip_ratio.load(config=self.config.get('clip_ratio', 0))

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

        self.returns = None
        self.advantages = None
        self.returns_stats = utils.IncrementalStatistics()

    def clear(self):
        pass

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

    def compute_returns(self, discount: float):
        """Computes the returns, also called rewards-to-go"""
        returns = utils.rewards_to_go(self.rewards, discount=discount)
        self.returns = self.returns_stats.update(returns, normalize=True)
        return returns

    def compute_advantages(self, gamma: float, lambda_: float):
        """Computes the advantages using generalized-advantage estimation"""
        mean = tf.cast(self.returns_stats.mean, dtype=tf.float32)
        std = tf.cast(self.returns_stats.std, dtype=tf.float32)
        denormalized_values = mean + self.values * std

        advantages = utils.gae(self.rewards, values=denormalized_values, gamma=gamma, lambda_=lambda_, normalize=True)
        self.advantages = tf.cast(advantages, dtype=tf.float32)

        return denormalized_values

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
