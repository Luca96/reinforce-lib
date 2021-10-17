
import tensorflow as tf

from tensorflow.keras.layers import *

from rl import utils
from rl.agents import Agent
from rl.layers import DuelingLayer

from rl.v2.networks import backbones, Network

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
        self._base_model_initialized = True
        self.has_prioritized_mem = bool(prioritized)

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
        self.gamma_n = tf.pow(x=self.agent.gamma, y=self.agent.horizon)

    @tf.function
    def call(self, inputs, actions=None, training=None):
        q_values = super().call(inputs, training=training)

        # TODO: consider to multiply q-values times one_hot(actions) + sum, instead of indexing
        if tf.is_tensor(actions):
            # actions = tf.one_hot(actions, self.agent.num_classes, 1.0, 0.0)
            # return tf.reduce_sum(q_values * actions, axis=1)

            # index q-values by given actions
            q_values = utils.index_tensor(tensor=q_values, indices=actions)
            return tf.expand_dims(q_values, axis=-1)

        return q_values

    @tf.function
    def q_values(self, inputs, **kwargs):
        return self(inputs, **kwargs)

    @tf.function
    def act(self, inputs):
        q_values = self(inputs, training=False)
        return tf.argmax(q_values, axis=-1)

    def structure(self, inputs: Dict[str, Input], name='Deep-Q-Network', **kwargs) -> tuple:
        utils.remove_keys(kwargs, keys=['dueling', 'operator', 'prioritized'])
        # inputs = inputs['state']
        #
        # if len(inputs.shape) <= 2:
        #     x = backbones.dense(layer=inputs, **kwargs)
        # else:
        #     x = backbones.convolutional(layer=inputs, **kwargs)
        #
        # output = self.output_layer(layer=x)
        # return inputs, output, name
        return super().structure(inputs, name=name, **kwargs)

    def output_layer(self, layer: Layer, **kwargs) -> Layer:
        assert self.agent.num_actions == 1

        if self.use_dueling:
            return self.dueling_architecture(layer)

        return Dense(units=self.agent.num_classes, name='q-values', **kwargs)(layer)

    # def dueling_architecture(self, layer: Layer) -> Layer:
    #     # two streams (branches)
    #     value = Dense(units=1, name='value', **self.output_args)(layer)
    #     advantage = Dense(units=self.agent.num_classes, name='advantage', **self.output_args)(layer)
    #
    #     if self.operator == 'max':
    #         k = tf.reduce_max(advantage, axis=-1, keepdims=True)
    #     else:
    #         k = tf.reduce_mean(advantage, axis=-1, keepdims=True)
    #
    #     q_values = value + (advantage - k)
    #     return q_values

    def dueling_architecture(self, layer: Layer) -> Layer:
        dueling = DuelingLayer(units=self.agent.num_classes, operator=self.operator.lower(), **self.output_args)
        return dueling(layer)

    @tf.function
    def objective(self, batch: dict, reduction=tf.reduce_mean) -> tuple:
        actions = batch['action']
        q_values = self(inputs=batch['state'], actions=actions, training=True)
        q_targets = self.targets(batch)
        td_error = q_targets - q_values

        if self.has_prioritized_mem:
            # inform agent's memory about td-error, to later update priorities
            self.agent.memory.td_error.assign(tf.stop_gradient(tf.squeeze(td_error)))

            loss = reduction(td_error * batch['_weights'])
        else:
            loss = reduction(self.loss_fn(td_error))

        debug = dict(loss=loss, targets=q_targets, values=q_values, returns=batch['return'],
                     td_error=td_error, values_hist=q_values, targets_hist=q_targets)

        # find q-values and targets of each action, separately
        for a in range(self.agent.num_classes):
            mask = actions == a

            if mask.shape[0] > 0:
                debug[f'value({a})'] = q_values[mask]
                debug[f'target({a})'] = q_targets[mask]
                debug[f'td-error({a})'] = td_error[mask]

        if '_weights' in batch:
            debug['weights_IS'] = batch['_weights']

        return loss, debug

    @tf.function
    def targets(self, batch: dict):
        """Computes target Q-values using target network"""
        returns = batch['return']
        q_values = self.target(inputs=batch['next_state'], training=False)

        targets = returns + self.gamma_n * (1.0 - batch['terminal']) * tf.reduce_max(q_values, axis=1, keepdims=True)
        return tf.stop_gradient(targets)


@Network.register()
class DoubleQNetwork(QNetwork):

    @tf.function
    def targets(self, batch: dict):
        returns = batch['return']
        next_states = batch['next_state']

        # double q-learning rule
        q_target = self(inputs=next_states, training=False)
        argmax_a = tf.expand_dims(tf.argmax(q_target, axis=-1), axis=-1)
        q_values = self.target(inputs=next_states, actions=argmax_a, training=False)

        targets = returns + self.gamma_n * (1.0 - batch['terminal']) * q_values
        return tf.stop_gradient(targets)


# @Network.register()
# class CategoricalQNetwork(QNetwork):
#
#     def __init__(self, agent: Agent, target=True, log_prefix='cqn', **kwargs):
#         utils.remove_keys(kwargs, keys=['dueling', 'operator', 'prioritized'])
#         self._base_model_initialized = True
#
#         self.num_classes = agent.num_classes
#         self.num_atoms = agent.num_atoms
#         self.support = agent.support
#         self.v_min = agent.v_min
#         self.v_max = agent.v_max
#
#         super().__init__(agent, target=target, log_prefix=log_prefix, **kwargs)
#
#     @tf.function
#     def call(self, inputs, actions=None, argmax=False, training=None):
#         support = self.support  # z_i ("atoms")
#         support_prob = super().call(inputs, training=training)  # p_i
#
#         if tf.is_tensor(actions):
#             # select support probabilities according to *given* actions
#             # return utils.index_tensor(support_prob, indices=actions)
#             return self.index(support_prob, actions)
#
#         q_values = tf.reduce_sum(support * support_prob, axis=1)
#
#         if argmax:
#             a_star = tf.argmax(q_values, axis=-1)
#
#             # select support probabilities according to *best* actions
#             # sp = utils.index_tensor(support_prob, indices=a_star)
#             return self.index(support_prob, a_star)
#             # return tf.expand_dims(prob, axis=-1)
#
#         return q_values
#
#     @staticmethod
#     @tf.function
#     def index(prob, actions):
#         batch_size, num_atoms = prob.shape[:2]
#
#         # prepare indices along..
#         idx_a = tf.repeat(actions, repeats=[num_atoms] * batch_size)  # action-dim
#         idx_z = tf.tile(tf.range(0, limit=num_atoms), multiples=[batch_size])  # atoms-dim
#         idx_b = tf.repeat(tf.range(0, limit=batch_size), repeats=[num_atoms] * batch_size)  # batch-dim
#
#         # concat and reshape to get final indices
#         indices = tf.stack([idx_b, idx_z, tf.cast(idx_a, dtype=tf.int32)], axis=1)
#         indices = tf.reshape(indices, shape=(batch_size * num_atoms, 3))
#
#         # finally retrieve stuff at indices
#         tensor = tf.gather_nd(prob, indices)
#         return tf.reshape(tensor, shape=(batch_size, num_atoms))
#
#     @tf.function
#     def act(self, inputs):
#         q_values = self(inputs, training=False)
#         return tf.argmax(q_values, axis=-1)
#
#     def structure(self, inputs: Dict[str, Input], name='Categorical-Q-Network', **kwargs) -> tuple:
#         return super().structure(inputs, name=name, **kwargs)
#
#     def output_layer(self, layer: Layer) -> Layer:
#         assert self.agent.num_actions == 1
#
#         out = Dense(units=self.num_atoms * self.num_classes, name='z-logits', **self.output_args)(layer)
#         out = Reshape((self.num_atoms, self.num_classes))(out)
#         return tf.nn.softmax(out, axis=1)  # per-action softmax
#
#     @tf.function
#     def objective(self, batch: dict, reduction=tf.reduce_mean) -> tuple:
#         prob = self(batch['state'], actions=batch['action'], training=True)
#         prob_next = self.target(batch['next_state'], argmax=True, training=False)
#
#         # compute projection on support
#         proj_z = tf.matmul(batch['reward'], tf.transpose(self.agent.gamma * self.support))
#         proj_z = tf.clip_by_value(proj_z, self.v_min, self.v_max)
#
#         b = (proj_z - self.v_min) / self.agent.delta
#
#         low = tf.math.floor(b)  # "low" and "upp" are indices
#         upp = tf.math.ceil(b)
#
#         # distribute probability of the projected support
#         m = tf.zeros(shape=(prob_next.shape[0], self.num_atoms))
#
#         prob_low = prob_next * (upp - b)
#         prob_upp = prob_next * (b - low)
#
#         # def scatter_add(args: tuple):
#         #     tensor, indices, updates = args
#         #     return tf.tensor_scatter_nd_add(tensor, tf.expand_dims(indices, axis=-1), updates)
#
#         m_low = tf.map_fn(fn=self._scatter_add, elems=(m, low, prob_low), fn_output_signature=tf.float32)
#         m_upp = tf.map_fn(fn=self._scatter_add, elems=(m, upp, prob_upp), fn_output_signature=tf.float32)
#
#         # cross-entropy loss
#         loss = tf.stop_gradient(m_low + m_upp) * tf.math.log(prob)
#         # loss = (m_low + m_upp) * tf.math.log(prob)
#
#         loss = tf.reduce_sum(loss, axis=1)  # sum over atoms (z_i)
#         loss = reduction(-loss)  # mean over batch of trajectories
#
#         return loss, {}  # dict(loss=loss, prob=prob, prob_target=prob_next, projected_support=proj_z)
#
#     @staticmethod
#     @tf.function
#     def _scatter_add(args: tuple):
#         tensor, indices, updates = args
#         indices = tf.cast(indices, dtype=tf.int32)
#
#         return tf.tensor_scatter_nd_add(tensor, tf.expand_dims(indices, axis=-1), updates)
#
#     def targets(self, batch: dict):
#         pass


@Network.register()
class RainbowQNetwork(QNetwork):
    # https://github.com/google/dopamine/blob/master/dopamine/agents/rainbow/rainbow_agent.py

    def __init__(self, agent: Agent, target=True, log_prefix='cqn', **kwargs):
        # utils.remove_keys(kwargs, keys=['dueling', 'operator', 'prioritized'])
        utils.remove_keys(kwargs, keys=['prioritized'])
        self._base_model_initialized = True

        self.num_classes = agent.num_classes
        self.num_atoms = agent.num_atoms
        self.support = agent.support
        self.v_min = agent.v_min
        self.v_max = agent.v_max

        super().__init__(agent, target=target, log_prefix=log_prefix, **kwargs)

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

    # def output_layer(self, layer: Layer) -> Layer:
    #     assert self.agent.num_actions == 1
    #
    #     logits = Dense(units=self.num_atoms * self.num_classes, name='z-logits', **self.output_args)(layer)
    #     logits = Reshape((self.num_classes, self.num_atoms))(logits)
    #     return logits

    def output_layer(self, layer: Layer) -> Layer:
        assert self.agent.num_actions == 1

        if self.use_dueling:
            return self.dueling_architecture(layer)

        logits = Dense(units=self.num_atoms * self.num_classes, name='z-logits', **self.output_args)(layer)
        logits = Reshape((self.num_classes, self.num_atoms))(logits)
        return logits

    def dueling_architecture(self, layer: Layer) -> Layer:
        # two streams (branches)
        value = Dense(units=self.num_atoms, name='value', **self.output_args)(layer)

        adv = Dense(units=self.num_atoms * self.agent.num_classes, name='advantage', **self.output_args)(layer)
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
            self.agent.memory.td_error.assign(loss)

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
    def call(self, states, num_quantiles: int, training=None, risk=False, **kwargs):
        quantiles = self._sample_quantiles(shape=(states.shape[0], num_quantiles))

        if risk:
            inputs = (states, self.beta_fn(quantiles))
        else:
            inputs = (states, quantiles)

        quantile_values = super().call(inputs, actions=None, training=training)

        return quantile_values, quantiles

    @tf.function
    def q_values(self, inputs, **kwargs):
        quantile_values, _ = self(inputs, num_quantiles=self.policy_samples, risk=True, **kwargs)
        return tf.reduce_mean(quantile_values, axis=0, keepdims=True)

    @tf.function
    def act(self, state):
        quantile_values, _ = self(state, num_quantiles=self.policy_samples, risk=True)
        q_values = tf.reduce_mean(quantile_values, axis=0)

        return tf.argmax(q_values, axis=0)

    # @tf.function
    def act_target(self, next_states):
        quantile_values, _ = self.target(next_states, num_quantiles=self.quantile_samples, risk=True)

        # action selection
        if self.double_dqn:
            z_values_actions, _ = self(next_states, num_quantiles=self.policy_samples)
        else:
            z_values_actions, _ = self.target(next_states, num_quantiles=self.policy_samples)

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

        if len(states.shape) <= 2:
            x = backbones.dense(layer=states, **kwargs)
        else:
            x = backbones.convolutional(layer=states, **kwargs)

        # x = backbones.dense(layer=states, **kwargs)
        x = Dense(units=self.embedding_size)(x)

        emb = self.embed_fn(quantiles)
        x = self.merge_fn(x, emb)
        x = Dense(units=kwargs.get('units', 64))(x)

        quantile_values = self.output_layer(layer=x)

        return [states, quantiles], quantile_values, name

    # @tf.function
    def embed_fn(self, quantiles: Layer) -> Layer:
        embedding = tf.tile(quantiles, multiples=(1, 1, self.embedding_size))

        pi = utils.TF_PI
        indices = tf.range(start=1, limit=self.embedding_size + 1, dtype=tf.float32)

        embedding = tf.cos(pi * indices * embedding)
        embedding = Dense(units=self.embedding_size, activation='relu', name='embed')(embedding)

        return embedding

    # @tf.function
    def merge_fn(self, x: Layer, y: Layer) -> Layer:
        x = tf.expand_dims(x, axis=1)
        x = Multiply(name='merge')([x, y])

        # shape: (batch_size * |num_quantiles|,  embedding_size)
        return tf.reshape(x, shape=(-1, x.shape[-1]))

    # @tf.function
    def objective(self, batch: dict, reduction=tf.reduce_mean) -> tuple:
        quantile_values, quantiles = self(batch['state'], num_quantiles=self.quantile_samples)
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

    # @tf.function
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
    def _sample_quantiles(self, shape: tuple) -> tf.Tensor:
        return tf.random.uniform(shape=shape + (1,), minval=0, maxval=1, dtype=tf.float32, seed=self.agent.seed)

    def get_inputs(self) -> Dict[str, Input]:
        inputs = super().get_inputs()

        # add sampled quantiles (tau) as Input layer
        inputs['quantile'] = Input(shape=(None, 1), name='quantile')
        return inputs
