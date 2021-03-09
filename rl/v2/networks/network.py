
import tensorflow as tf

from rl import utils
from rl.agents import Agent
from rl.parameters import DynamicParameter

from typing import Union, List, Dict


# TODO: 'custom' metrics?
# TODO: distributed strategy
class Network(tf.keras.Model):
    """Base class for agent's networks (e.g. Q, Policy, Value, ...)"""

    def __init__(self, agent: Agent, target=False, log_prefix='network', **kwargs):
        self.agent = agent
        self.prefix = str(log_prefix)
        self.output_args = kwargs.pop('output', {})

        # Optimization:
        self.optimizer = None
        self.clip_norm = None
        self.should_clip_gradients = None

        # Model:
        inputs, outputs, name = self.structure(inputs=self.get_inputs(), **kwargs)

        super().__init__(inputs=inputs, outputs=outputs, name=name)

        # Target network:
        if target:
            self.target = TargetNetwork(self.__class__(agent, target=False, **kwargs))
            self.target.set_weights(weights=self.get_weights())

    def compile(self, optimizer: str, clip_norm: utils.DynamicType = None, **kwargs):
        self.optimizer = utils.get_optimizer_by_name(optimizer, **kwargs)

        if clip_norm is None:
            self.should_clip_gradients = False
        else:
            self.should_clip_gradients = True
            self.clip_norm = DynamicParameter.create(value=clip_norm)  # TODO: create in agent

        super().compile(optimizer=self.optimizer, loss=self.objective, run_eagerly=True)

    def structure(self, **kwargs) -> tuple:
        """Specified the network's structure (i.e. layers)"""
        raise NotImplementedError

    def output_layer(self, *args, **kwargs) -> tf.keras.layers.Layer:
        raise NotImplementedError

    def objective(self, batch, reduction=tf.reduce_mean) -> tuple:
        raise NotImplementedError

    def update_target_network(self, copy_weights=False, polyak=0.995):
        if copy_weights:
            self.target.set_weights(weights=self.get_weights())
        else:
            utils.polyak_averaging2(model=self, target=self.target, alpha=polyak)

    def train_step(self, batch: dict):
        if isinstance(batch, tuple):
            batch = batch[0]

        loss, debug = self.train_on_batch(batch)

        self.agent.log(average=True, **({f'{self.prefix}_{k}': v for k, v in debug.items()}))

        return dict(loss=loss, gradient_norm=tf.reduce_mean(debug['gradient_norm']))

    @tf.function
    def train_on_batch(self, batch):
        with tf.GradientTape() as tape:
            loss, debug = self.objective(batch)

        trainable_vars = self.trainable_variables

        gradients = tape.gradient(loss, trainable_vars)
        debug['gradient_norm'] = [tf.norm(g) for g in gradients]

        if self.should_clip_gradients:
            # gradients = utils.clip_gradients(gradients, norm=self.clip_norm())
            # debug['gradient_clipped_norm'] = [tf.norm(g) for g in gradients]
            # debug['clip_norm'] = self.clip_norm.value
            gradients = self.clip_gradients(gradients, debug)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return loss, debug

    def clip_gradients(self, gradients: List[tf.Tensor], debug: dict) -> List[tf.Tensor]:
        gradients, global_norm = utils.clip_gradients2(gradients, norm=self.clip_norm())
        debug['gradient_clipped_norm'] = [tf.norm(g) for g in gradients]
        debug['gradient_global_norm'] = global_norm
        debug['clip_norm'] = self.clip_norm.value
        return gradients

    def get_inputs(self) -> Dict[str, tf.keras.layers.Input]:
        """Transforms an arbitrary complex state-space as `tf.keras.layers.Input` layers"""
        inputs = dict()

        for name, shape in self.agent.state_spec.items():
            inputs[name] = tf.keras.layers.Input(shape=shape, dtype=tf.float32, name=name)

        return inputs

    def get_config(self):
        pass


class TargetNetwork:
    """A 'proxy' that wraps a `Network` instance.
        - Such wrapping is necessary since 'target' networks are created within their class, so this
          prevents tf.keras.Model to track their weights, which causes lots of annoyance.
    """

    def __init__(self, network: Network):
        self.network = network
        self.network._name = f'{self.network.name}-Target'

    def __call__(self, *args, **kwargs):
        return self.network(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.network, name)
