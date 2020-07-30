
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from typing import List, Union, Dict
from rl import utils


class Network:
    pass


# TODO: disentangle policy-net from value-net, so that each of them can be arbitrary subclassed, moreover a
#  Network class can be composed by these policy/value/Q-network classes...
class PPONetwork(Network):
    def __init__(self, agent, **kwargs):
        from rl.agents.ppo import PPOAgent

        self.agent: PPOAgent = agent
        self.distribution = self.agent.distribution_type
        self.mixture_components = self.agent.mixture_components

        # policy and value networks
        self.policy = self.policy_network(**kwargs)
        self.value = self.value_network(**kwargs)

    @tf.function
    def predict(self, inputs: Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]]):
        policy = self.policy(inputs, training=False)

        if self.distribution != 'categorical':
            # round samples (actions) before computing density:
            # motivation: https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Beta
            log_prob = policy.log_prob(tf.clip_by_value(policy, utils.EPSILON, 1.0 - utils.EPSILON))
            mean = policy.mean()
            std = policy.stddev()
        else:
            mean = 0.0
            std = 0.0
            log_prob = policy.log_prob(policy)

        value = self.value(inputs, training=False)

        return policy, mean, std, log_prob, value

    @tf.function
    def act(self, inputs: Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]]):
        action = self.policy(inputs, training=False)
        return action

    def update_step_policy(self, batch):
        with tf.GradientTape() as tape:
            loss, kl = self.agent.policy_objective(batch)

        gradients = tape.gradient(loss, self.policy.trainable_variables)

        if self.agent.should_clip_policy_grads:
            gradients = [tf.clip_by_norm(grad, clip_norm=self.agent.grad_norm_policy) for grad in gradients]

        self.agent.policy_optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))

        return loss, kl, gradients

    def update_step_value(self, batch):
        with tf.GradientTape() as tape:
            loss = self.agent.value_objective(batch)

        gradients = tape.gradient(loss, self.value.trainable_variables)

        if self.agent.should_clip_value_grads:
            gradients = [tf.clip_by_norm(grad, clip_norm=self.agent.grad_norm_value) for grad in gradients]

        self.agent.value_optimizer.apply_gradients(zip(gradients, self.value.trainable_variables))

        return loss, gradients

    def policy_layers(self, inputs: Dict[str, Input], **kwargs):
        """Defines the architecture of the policy-network"""
        units = kwargs.get('units', 32)
        num_layers = kwargs.get('num_layers', kwargs.get('layers', 2))  # 'num_layers' or 'layers'
        activation = kwargs.get('activation', tf.nn.swish)
        dropout_rate = kwargs.get('dropout', 0.0)
        linear_units = kwargs.get('linear_units', 0)

        x = Dense(units, activation=activation)(inputs['state'])
        x = LayerNormalization()(x)

        for _ in range(0, num_layers, 2):
            if dropout_rate > 0.0:
                x = Dense(units, activation=activation)(x)
                x = Dropout(rate=dropout_rate)(x)

                x = Dense(units, activation=activation)(x)
                x = Dropout(rate=dropout_rate)(x)
            else:
                x = Dense(units, activation=activation)(x)
                x = Dense(units, activation=activation)(x)

            x = LayerNormalization()(x)

        if linear_units > 0:
            x = Dense(units=linear_units, activation='linear')(x)

        return x

    def value_layers(self, inputs: Dict[str, Input], **kwargs):
        """Defines the architecture of the value-network"""
        return self.policy_layers(inputs, **kwargs)

    def policy_network(self, **kwargs):
        inputs = self._get_input_layers()
        last_layer = self.policy_layers(inputs, **kwargs)
        action = self.get_distribution_layer(last_layer)

        return Model(list(inputs.values()), outputs=action, name='Policy-Network')

    def value_network(self, **kwargs):
        inputs = self._get_input_layers()
        last_layer = self.value_layers(inputs, **kwargs)
        value = Dense(units=1, activation=None, dtype=tf.float32, name='value_head')(last_layer)

        return Model(list(inputs.values()), outputs=value, name='Value-Network')

    # def get_distribution_layer(self, layer: Layer) -> tfp.layers.DistributionLambda:
    #     if self.distribution_type == 'categorical':
    #         num_params = tfp.layers.CategoricalMixtureOfOneHotCategorical.params_size(
    #             event_size=self.action_shape[1], num_components=self.mixture_components)
    #
    #         layer = Dense(units=num_params, activation=None, name='logits')(layer)
    #
    #         return tfp.layers.CategoricalMixtureOfOneHotCategorical(
    #             event_size=self.action_shape, num_components=self.mixture_components, sample_dtype=tf.int32)(layer)
    #
    #     if self.distribution_type == 'beta':
    #         raise NotImplementedError
    #
    #     if self.distribution_type == 'gaussian':
    #         num_params = tfp.layers.MixtureNormal.params_size(self.mixture_components, event_shape=self.action_shape)
    #
    #         layer = Dense(units=num_params, activation=None)(layer)
    #
    #         return tfp.layers.MixtureNormal(self.mixture_components, event_shape=self.action_shape)(layer)

    def get_distribution_layer(self, layer: Layer) -> tfp.layers.DistributionLambda:
        # Discrete actions:
        if self.distribution == 'categorical':
            num_actions = self.agent.num_actions
            num_classes = self.agent.num_classes

            if self.mixture_components == 1:
                logits = Dense(units=num_actions * num_classes, activation='linear', name='logits')(layer)

                if num_actions > 1:
                    logits = Reshape((num_actions, num_classes))(logits)
                else:
                    logits = tf.expand_dims(logits, axis=0)

                return tfp.layers.DistributionLambda(
                    make_distribution_fn=lambda t: tfp.distributions.Categorical(logits=t))(logits)
            else:
                return utils.get_mixture_of_categorical(layer, num_actions=num_classes,
                                                        num_components=self.mixture_components)
        # Bounded continuous 1-dimensional actions:
        # for activations choice refer to chapter 4 of http://proceedings.mlr.press/v70/chou17a/chou17a.pdf
        if self.distribution == 'beta':
            num_actions = self.agent.num_actions

            if self.mixture_components == 1:
                # make a, b > 1, so that the Beta distribution is concave and unimodal (see paper above)
                # alpha = Dense(units=self.num_actions, activation=utils.softplus_one, name='alpha')(layer)
                # beta = Dense(units=self.num_actions, activation=utils.softplus_one, name='beta')(layer)

                # TODO: seems more stable than above, but still not enough...
                alpha = Dense(units=num_actions, activation='softplus', name='alpha')(layer)
                beta = Dense(units=num_actions, activation='softplus', name='beta')(layer)

                return tfp.layers.DistributionLambda(
                    make_distribution_fn=lambda t: tfp.distributions.Beta(t[0], t[1]))([alpha, beta])
            else:
                return utils.get_mixture_of_beta(layer, num_actions, num_components=self.mixture_components)

        # Unbounded continuous actions)
        # for activations choice see chapter 4 of http://proceedings.mlr.press/v70/chou17a/chou17a.pdf
        if self.distribution == 'gaussian':
            num_actions = self.agent.num_actions

            if self.mixture_components == 1:
                mu = Dense(units=num_actions, activation='linear', name='mu')(layer)
                sigma = Dense(units=num_actions, activation='softplus', name='sigma')(layer)

                # ensure variance > 0, so that loss doesn't diverge or became NaN
                sigma = tf.add(sigma, utils.EPSILON)

                return tfp.layers.DistributionLambda(
                    make_distribution_fn=lambda t: tfp.distributions.Normal(loc=t[0], scale=t[1])
                )([mu, sigma])
            else:
                event_shape = [num_actions]
                num_params = tfp.layers.MixtureNormal.params_size(num_components=self.mixture_components,
                                                                  event_shape=event_shape)

                layer = Dense(units=num_params, activation=None)(layer)
                return tfp.layers.MixtureNormal(num_components=self.mixture_components, event_shape=event_shape)(layer)

    def reset(self):
        pass

    def load_weights(self):
        self.policy.load_weights(filepath=self.agent.weights_path['policy'], by_name=False)
        self.value.load_weights(filepath=self.agent.weights_path['value'], by_name=False)

    def save_weights(self):
        self.policy.save_weights(filepath=self.agent.weights_path['policy'])
        self.value.save_weights(filepath=self.agent.weights_path['value'])

    def summary(self):
        print('==== Policy Network ====')
        self.policy.summary()

        print('\n==== Value Network ====')
        self.value.summary()

    # def register(self, **kwargs):
    #     """Registers tensors as tf.keras's Input layers"""
    #     for name, shape in kwargs.items():
    #         assert isinstance(name, str)
    #         assert isinstance(shape, tuple)
    #
    #         self.inputs[name] = Input(shape=shape, dtype=tf.float32, name=name)
    #
    # def retrieve(self, names: Union[str, List[str]]):
    #     """Retrieves input tensors by name"""
    #     if isinstance(names, str):
    #         return self.inputs[names]
    #
    #     return [self.inputs[name] for name in names]

    def _get_input_layers(self) -> Dict[str, Input]:
        """Handles arbitrary complex state-spaces"""
        input_layers = dict()

        for name, shape in self.agent.state_spec.items():
            if self.agent.drop_batch_reminder:
                layer = Input(shape=shape, batch_size=self.agent.batch_size, dtype=tf.float32, name=name)
            else:
                layer = Input(shape=shape, dtype=tf.float32, name=name)

            input_layers[name] = layer

        return input_layers
