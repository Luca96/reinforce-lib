
import tensorflow as tf

from tensorflow.keras.layers import *

from rl import utils
from rl.agents import Agent
from rl.layers import DuelingLayer

from rl.networks import backbones, Network

from functools import partial
from typing import Dict, Union, Callable


@Network.register(name='Q-Network')
class QNetwork(Network):
    """Deep Q-Network (DQN) with support for:
        - `target` network,
        - `dueling` architecture, and
        - `prioritized` memory.
    """

    def __init__(self, agent: Agent, target=True, dueling=False, operator='avg', log_prefix='q', prioritized=False,
                 loss: Union[str, float, callable] = 'mse', **kwargs):
        self.init_hack()
        self.has_prioritized_mem = bool(prioritized)

        self.num_actions = agent.action_converter.num_actions
        self.num_classes = agent.action_converter.num_classes

        # choose loss function (huber, mse, or user-defined)
        if isinstance(loss, (int, float)):
            self.loss_fn = partial(utils.huber_loss, kappa=tf.constant(loss, dtype=tf.float32))

        elif callable(loss):
            self.loss_fn = loss
        else:
            self.loss_fn = tf.function(lambda x: 0.5 * tf.square(x))  # mse

        if dueling:
            assert isinstance(operator, str) and operator.lower() in ['avg', 'max']
            self.operator = operator.lower()
            self.use_dueling = True
        else:
            self.use_dueling = False

        super().__init__(agent, target=target, dueling=dueling, operator=operator, log_prefix=log_prefix, **kwargs)

        # cumulative gamma
        self.gamma_n = tf.pow(x=self.agent.gamma, y=getattr(self.agent, 'horizon', 1))

    @property
    def default_name(self) -> str:
        return 'Deep-Q-Network'

    # TODO: @tf.function?
    def call(self, inputs, actions=None, training=None, **kwargs):
        q_values = super().call(inputs, training=training, **kwargs)

        if utils.is_tensor_like(actions):
            # index q-values by given actions
            q_values = utils.index_tensor(tensor=q_values, indices=actions)
            return tf.expand_dims(q_values, axis=-1)

        return q_values

    @tf.function
    def q_values(self, inputs, **kwargs):
        return self(inputs, **kwargs)

    @tf.function
    def act(self, inputs, **kwargs):
        q_values = self(inputs, training=False, **kwargs)
        return tf.argmax(q_values, axis=-1)

    def structure(self, inputs: Dict[str, Input], **kwargs) -> tuple:
        utils.remove_keys(kwargs, keys=['dueling', 'operator', 'prioritized'])
        return super().structure(inputs, **kwargs)

    def output_layer(self, layer: Layer, **kwargs) -> Layer:
        assert self.num_actions == 1

        if self.use_dueling:
            return self.dueling_architecture(layer, **kwargs)

        return Dense(units=self.num_classes, name='q-values', **kwargs)(layer)

    def dueling_architecture(self, layer: Layer, **kwargs) -> Layer:
        dueling = DuelingLayer(units=self.num_classes, operator=self.operator.lower(), **kwargs)
        return dueling(layer)

    @tf.function
    def objective(self, batch: dict, reduction=tf.reduce_mean) -> tuple:
        actions = batch['action']
        q_values = self(inputs=batch['state'], actions=actions, training=True)
        q_targets = self.targets(batch)
        td_error = q_values - q_targets

        if self.has_prioritized_mem:
            # inform agent's memory about td-error, to later update priorities
            self.agent.memory.td_error.assign(tf.stop_gradient(tf.squeeze(td_error)))

            loss = reduction(self.loss_fn(td_error * batch['_weights']))
        else:
            loss = reduction(self.loss_fn(td_error))

        debug = dict(loss=loss, targets=q_targets, values=q_values, returns=batch['return'],
                     td_error=td_error, values_hist=q_values, targets_hist=q_targets)

        # find q-values and targets of each action, separately
        for a in range(self.num_classes):
            mask = actions == a

            if mask.shape[0] > 0:
                debug[f'value({a})'] = q_values[mask]
                debug[f'target({a})'] = q_targets[mask]
                debug[f'td-error({a})'] = td_error[mask]

        if '_weights' in batch:
            debug['weights_IS'] = batch['_weights']
            debug['td_error_weighted'] = tf.stop_gradient(td_error * batch['_weights'])

        return loss, debug

    @tf.function
    def targets(self, batch: dict):
        """Computes target Q-values using target network"""
        returns = batch['return']
        q_values = self.target(inputs=batch['next_state'], training=False)

        # TODO: SARSA uses `Q(s_t+1, a_t+1)` instead of `max_a Q(s_t+1, a)`; see GRL page 187
        # here we use the "returns" for the general case of n-step returns. When n=1, returns=rewards.
        targets = returns + self.gamma_n * (1.0 - batch['terminal']) * tf.reduce_max(q_values, axis=1, keepdims=True)
        return tf.stop_gradient(targets)


@Network.register()
class DoubleQNetwork(QNetwork):

    @property
    def default_name(self) -> str:
        return self.__class__.__name__

    @tf.function
    def targets(self, batch: dict):
        returns = batch['return']
        next_states = batch['next_state']

        # double q-learning rule
        q_next_a = self(inputs=next_states, training=False)
        argmax_a = tf.expand_dims(tf.argmax(q_next_a, axis=-1), axis=-1)
        q_values = self.target(inputs=next_states, actions=argmax_a, training=False)

        targets = returns + self.gamma_n * (1.0 - batch['terminal']) * q_values
        return tf.stop_gradient(targets)


@Network.register()
class RainbowQNetwork(QNetwork):
    # Based on: https://github.com/google/dopamine/blob/master/dopamine/agents/rainbow/rainbow_agent.py

    def __init__(self, agent: Agent, target=True, log_prefix='cqn', **kwargs):
        utils.remove_keys(kwargs, keys=['prioritized'])
        self.init_hack()

        self.num_atoms = agent.num_atoms
        self.support = agent.support
        self.v_min = agent.v_min
        self.v_max = agent.v_max

        super().__init__(agent, target=target, log_prefix=log_prefix, **kwargs)

    @property
    def default_name(self) -> str:
        return 'Rainbow-Q-Network'

    @tf.function
    def call(self, inputs, training=None, **kwargs):
        logits = super().call(inputs, training=training)  # p = (B, |A|, |Z|)
        probabilities = tf.nn.softmax(logits)

        q_values = tf.reduce_sum(self.support * probabilities, axis=2)

        return q_values, logits, probabilities

    @tf.function
    def q_values(self, inputs, **kwargs):
        q_values, _, _ = self(inputs, training=False)
        return q_values

    @tf.function
    def act(self, inputs, **kwargs):
        q_values = self.q_values(inputs, **kwargs)
        return tf.argmax(q_values, axis=-1)

    def output_layer(self, layer: Layer, **kwargs) -> Layer:
        assert self.num_actions == 1

        if self.use_dueling:
            return self.dueling_architecture(layer, **kwargs)

        logits = Dense(units=self.num_atoms * self.num_classes, name='z-logits', **kwargs)(layer)
        logits = Reshape((self.num_classes, self.num_atoms))(logits)
        return logits

    def dueling_architecture(self, layer: Layer, **kwargs) -> Layer:
        # two streams (branches)
        value = Dense(units=self.num_atoms, name='value', **kwargs)(layer)

        adv = Dense(units=self.num_atoms * self.num_classes, name='advantage', **kwargs)(layer)
        adv = Reshape((self.num_classes, self.num_atoms))(adv)

        # reduce on action-dimension (axis 1)
        if self.operator == 'max':
            k = tf.reduce_max(adv, axis=1, keepdims=True)
        else:
            k = tf.reduce_mean(adv, axis=1, keepdims=True)

        # expand action dims to allow broadcasting. Shape: (batch, actions, atoms)
        return value[:, None, :] + (adv - k[:, None, :])

    @tf.function
    def objective(self, batch: dict, reduction=tf.reduce_mean) -> tuple:
        q_values, logits, _ = self(batch['state'], training=True)
        target_distribution, _debug = self.targets(batch)

        indices = tf.range(self.agent.batch_size, dtype=tf.float32)[:, None]
        actions = tf.concat([indices, batch['action']], axis=1)
        action_logits = tf.gather_nd(logits, indices=tf.cast(actions, dtype=tf.int32))

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=target_distribution, logits=action_logits)

        if self.has_prioritized_mem:
            self.agent.memory.td_error.assign(tf.stop_gradient(loss))

        loss = reduction(loss)

        debug = dict(loss=loss, q_values=q_values, target_distribution=target_distribution, logits=logits)
        debug.update(**_debug)

        return loss, debug

    @tf.function
    def targets(self, batch: dict):
        batch_size = self.agent.batch_size

        # shape: (batch_size, num_atoms)
        tiled_support = tf.tile(self.support, multiples=[batch_size])
        tiled_support = tf.reshape(tiled_support, shape=(batch_size, self.num_atoms))

        gamma_with_terminal = self.gamma_n * (1.0 - batch['terminal'])
        target_support = batch['return'] + gamma_with_terminal * tiled_support

        next_q, _, prob = self.target(batch['next_state'], training=False)
        batch_indices = tf.range(batch_size, dtype=tf.int64)[:, None]

        next_q_argmax = tf.argmax(next_q, axis=1)[:, None]
        next_q_argmax_indices = tf.concat([batch_indices, next_q_argmax], axis=1)

        next_prob = tf.gather_nd(prob, indices=next_q_argmax_indices)

        projection = self.project_distribution(target_support, next_prob, self.support)
        return tf.stop_gradient(projection), dict(q_targets=next_q, next_prob=next_prob)

    @tf.function
    def project_distribution(self, support, weights, target_support):
        target_deltas = target_support[1:] - target_support[:-1]
        delta_z = target_deltas[0]

        batch_size = self.agent.batch_size
        num_dims = tf.shape(target_support)[0]

        clipped_support = tf.clip_by_value(support, self.v_min, self.v_max)[:, None, :]
        tiled_support = tf.tile([clipped_support], multiples=(1, 1, num_dims, 1))

        target_support = tf.tile(target_support[:, None], multiples=(batch_size, 1))
        target_support = tf.reshape(target_support, shape=(batch_size, num_dims, 1))

        numerator = tf.abs(tiled_support - target_support)
        quotient = 1.0 - (numerator / delta_z)

        clipped_quotient = tf.clip_by_value(quotient, 0.0, 1.0)
        inner_prod = clipped_quotient * weights[:, None, :]

        projection = tf.reduce_sum(inner_prod, axis=3)
        return tf.reshape(projection, shape=(batch_size, num_dims))


@Network.register()
class ImplicitQuantileNetwork(QNetwork):
    # https://github.com/google/dopamine/blob/b312d26305222d676f84d8949e2f07763b63ea65/dopamine/discrete_domains/atari_lib.py#L249

    def __init__(self, *args, embedding_size=64, double=False, distortion_measure: Callable = None, **kwargs):
        assert embedding_size >= 1

        if distortion_measure is None:
            # the "distortion risk measure" is denoted as "beta" in the paper
            self.beta_fn = lambda x: x
        else:
            assert callable(distortion_measure)
            eta = kwargs.pop('eta', None)

            if isinstance(eta, (float, int)):
                self.beta_fn = lambda x: distortion_measure(x, eta=eta)
            else:
                self.beta_fn = lambda x: distortion_measure(x)  # use default `eta`

        self.embedding_size = int(embedding_size)
        self.quantile_shape: tuple = None
        self.double_dqn = bool(double)

        super().__init__(*args, **kwargs)

        self.policy_samples = self.agent.policy_samples  # K
        self.quantile_samples = self.agent.quantile_samples  # N = N'
        self.batch_size = self.agent.batch_size
        self.num_classes = self.agent.num_classes

    @tf.function
    def call(self, inputs: tuple, training=None, risk=False, **kwargs):
        states, quantiles = inputs

        if risk:
            inputs = (states, self.beta_fn(quantiles))
        else:
            inputs = (states, quantiles)

        return super().call(inputs, actions=None, training=training)

    @tf.function
    def q_values(self, states, **kwargs):
        quantiles = self._sample_quantiles(states)
        quantile_values = self((states, quantiles), risk=True, **kwargs)

        return tf.reduce_mean(quantile_values, axis=0, keepdims=True)

    @tf.function
    def act(self, state):
        quantiles = self._sample_quantiles(state)
        quantile_values = self((state, quantiles), risk=True)
        q_values = tf.reduce_mean(quantile_values, axis=0)

        return tf.argmax(q_values, axis=0)

    @tf.function
    def act_target(self, next_states):
        quantiles = self._sample_quantiles(next_states, num=self.quantile_samples)
        quantile_values = self.target((next_states, quantiles),  risk=True)

        # action selection
        inputs = (next_states, self._sample_quantiles(next_states))

        if self.double_dqn:
            z_values_actions = self(inputs)
        else:
            z_values_actions = self.target(inputs)

        z_values_actions = tf.reshape(z_values_actions,
                                      shape=(self.policy_samples, self.batch_size, self.num_classes))

        # compute Q-values and best actions
        q_values = tf.squeeze(tf.reduce_mean(z_values_actions, axis=0))
        a_argmax = tf.argmax(q_values, axis=1)

        return quantile_values, a_argmax

    def structure(self, inputs: Dict[str, Input], name='IQN', **kwargs) -> tuple:
        utils.remove_keys(kwargs, keys=['dueling', 'operator', 'prioritized'])

        states = inputs['state']
        quantiles = inputs['quantile']

        preproc_in = self.apply_preprocessing(inputs, preprocess=kwargs.pop('preprocess', None))

        if len(states.shape) <= 2:
            x = backbones.dense(layer=preproc_in['state'], **kwargs)
        else:
            x = backbones.convolutional(layer=preproc_in['state'], **kwargs)

        x = Dense(units=self.embedding_size)(x)

        emb = self.embed_fn(preproc_in['quantile'])
        x = self.merge_fn(x, emb)
        x = Dense(units=kwargs.get('units', 64))(x)

        quantile_values = self.output_layer(layer=x)

        return [states, quantiles], quantile_values

    def embed_fn(self, quantiles: Layer) -> Layer:
        embedding = tf.tile(quantiles, multiples=(1, 1, self.embedding_size))

        pi = utils.TF_PI
        indices = tf.range(start=1, limit=self.embedding_size + 1, dtype=tf.float32)

        embedding = tf.cos(pi * indices * embedding)
        embedding = Dense(units=self.embedding_size, activation='relu', name='embed')(embedding)

        return embedding

    def merge_fn(self, x: Layer, y: Layer) -> Layer:
        x = tf.expand_dims(x, axis=1)
        x = Multiply(name='merge')([x, y])

        # shape: (batch_size * |num_quantiles|,  embedding_size)
        return tf.reshape(x, shape=(-1, x.shape[-1]))

    @tf.function
    def objective(self, batch: dict, reduction=tf.reduce_mean) -> tuple:
        quantiles = self._sample_quantiles(batch['state'], num=self.quantile_samples)
        quantile_values = self((batch['state'], quantiles))
        target_quantile_values = self.targets(batch)

        # indices to index the quantiles
        indices = tf.range(self.quantile_samples * self.batch_size)
        indices = tf.expand_dims(indices, axis=-1)

        actions = tf.tile(batch['action'], multiples=(self.quantile_samples, 1))
        actions = tf.concat([indices, tf.cast(actions, dtype=indices.dtype)], axis=1)  # (N x batch_size, 2)

        # index, reshape, and transpose to (batch_size, N, 1)
        chosen_quantile_values = tf.gather_nd(quantile_values, indices=actions)
        chosen_quantile_values = tf.reshape(chosen_quantile_values,
                                            shape=(self.quantile_samples, self.batch_size, 1))
        chosen_quantile_values = tf.transpose(chosen_quantile_values, perm=[1, 0, 2])

        # Bellman error, and Huber loss; shape: (batch_size, N, N', 1)
        bellman_errors = target_quantile_values[:, :, None, :] - chosen_quantile_values[:, None, :, :]
        huber_loss = utils.huber_loss(bellman_errors)

        # reshape, transpose, and tile quantiles: (batch_size, N, N', 1)
        quantiles = tf.reshape(quantiles, shape=(self.quantile_samples, self.batch_size, 1))
        quantiles = tf.transpose(quantiles, perm=[1, 0, 2])
        quantiles = tf.tile(quantiles[:, None, :, :], multiples=(1, self.quantile_samples, 1, 1))

        # quantile Huber loss: (batch_size, N, N', 1)
        bool_bellman_errors = tf.stop_gradient(utils.to_float(bellman_errors < 0.0))
        quantile_huber_loss = (tf.abs(quantiles - bool_bellman_errors) * huber_loss) / self.agent.kappa()

        # sum over quantile-dimension (N), and average over target-quantiles (N')
        loss = tf.reduce_sum(quantile_huber_loss, axis=2)
        loss = tf.reduce_mean(loss)

        # debug
        debug = dict(loss=loss, quantile_huber_loss=quantile_huber_loss, huber_loss=huber_loss,
                     quantiles=quantiles, bellman_errors=bellman_errors, quantile_values=quantile_values,
                     target_quantile_values=target_quantile_values, chosen_quantile_values=chosen_quantile_values)

        return loss, debug

    @tf.function
    def targets(self, batch: dict) -> tf.Tensor:
        quantile_values, a_argmax = self.act_target(batch['next_state'])

        # reshape rewards to (N' x batch_size, 1)
        rewards = tf.tile(batch['reward'], multiples=(self.quantile_samples, 1))

        # discount factor: (N' x batch_size, 1)
        discount = self.agent.gamma * (1.0 - batch['terminal'])
        discount = tf.tile(discount, multiples=(self.quantile_samples, 1))

        # get indices associated to maximal Q-values
        a_argmax = tf.tile(a_argmax[:, None], multiples=(self.quantile_samples, 1))
        indices = tf.range(self.quantile_samples * self.batch_size, dtype=tf.int64)

        indexed_values = tf.concat([indices[:, None], a_argmax], axis=1)  # (N' x batch_size, 2)
        target_values = tf.gather_nd(quantile_values, indices=indexed_values)

        # shape: (N' x batch_size, 1)
        target_values = tf.expand_dims(target_values, axis=-1)
        target_values = tf.stop_gradient(rewards + discount * target_values)

        # reshape, transpose (batch_size, N', 1)
        target_values = tf.reshape(target_values, shape=(self.quantile_samples, self.batch_size, 1))
        target_values = tf.transpose(target_values, perm=[1, 0, 2])
        return target_values

    @tf.function
    def _sample_quantiles(self, states: tf.Tensor, num: int = None) -> tf.Tensor:
        shape = (states.shape[0], num or self.policy_samples)
        return tf.random.uniform(shape=shape + (1,), minval=0, maxval=1, dtype=tf.float32, seed=self.agent.seed)

    def get_inputs(self) -> Dict[str, Input]:
        inputs = super().get_inputs()

        # add sampled quantiles (tau) as Input layer
        inputs['quantile'] = Input(shape=(None, 1), name='quantile')
        return inputs
