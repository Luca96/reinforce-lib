
import tensorflow as tf

from rl import utils
from rl.v2.agents import Agent
from rl.parameters import DynamicParameter

from typing import Union, List, Dict, Callable


# TODO: check each `tf.is_tensor(x)` add `isinstance(x, dict)`
# TODO: monitor the "distance" (i.e. global_norm) of the network w.r.t its target (to debug polyak)
# TODO: distributed strategy
class Network(tf.keras.Model):
    """Base class for agent's networks (e.g. Q, Policy, Value, ...)"""
    _REGISTRY = utils.Registry()

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

    @staticmethod
    def create(*args, base_class: Union[str, Callable] = None, **kwargs) -> 'Network':
        """Instantiates a subclass of Network"""
        class_or_name = kwargs.pop('class', kwargs.pop('cls', None))

        if class_or_name is None:
            # use `base_class`
            assert base_class is not None

            if isinstance(base_class, str):
                base_class = Network._REGISTRY.retrieve(name=base_class)

            elif not callable(base_class):
                raise ValueError(f'Parameter "base_class" should be `str` or `callable` not {type(base_class)}.')

            class_ = base_class

        elif isinstance(class_or_name, str):
            class_ = Network._REGISTRY.retrieve(name=class_or_name)

        elif callable(class_or_name):
            class_ = class_or_name
        else:
            raise ValueError(f'Provided "class" should be `str` or `callable` not {type(class_or_name)}.')

        if base_class is None:
            base_class = Network

        elif isinstance(base_class, str):
            base_class = Network._REGISTRY.retrieve(name=base_class)

        elif not callable(base_class):
            raise ValueError(f'Parameter "base_class" should be `str` or `callable` not {type(base_class)}.')

        instance = class_(*args, **kwargs)

        assert isinstance(instance, base_class)
        return instance

    @staticmethod
    def register(name: str = None):
        # Based on: https://stackoverflow.com/questions/5929107/decorators-with-parameters

        def decorator(cls):
            assert issubclass(cls, Network)

            if '__main__' in str(cls) or '__mp_main__' in str(cls):
                # This avoid some weird errors if there is a __main__ guard in the same module where the network
                # (i.e. `cls`) is defined => @Network.register gets triggered more than once, thus causing troubles.
                # Also, `cls` is seen as defined two times, i.e:
                #   - rl.agents.<some>.cls (first trigger: OK),
                #   - __main__.cls (another trigger: WHY?!)
                return

            Network._REGISTRY.register(name or cls.__name__, class_=cls)
            return cls

        return lambda cls: decorator(cls)

    @classmethod
    def print_registered_networks(cls):
        print('Registered Networks [')
        spaces = max([len(name) for name, _ in cls._REGISTRY.registered_classes()])

        for name, class_ in cls._REGISTRY.registered_classes():
            print(f'\t{name:{spaces}} :: {class_}')

        print(']')

    def compile(self, optimizer: Union[str, dict], clip_norm: utils.DynamicType = None, **kwargs):
        if isinstance(optimizer, dict):
            name = optimizer.get('name', 'adam')

            kwargs.update(optimizer)  # add the remaining arguments if any
            kwargs.pop('name')
        else:
            name = str(optimizer)

        self.optimizer = utils.get_optimizer_by_name(name, **kwargs)

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
        debug['gradient_norm'] = utils.tf_norm(gradients)
        debug['gradient_global_norm'] = utils.tf_global_norm(debug['gradient_norm'])

        if self.should_clip_gradients:
            gradients = self.clip_gradients(gradients, debug)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return loss, debug

    def clip_gradients(self, gradients: List[tf.Tensor], debug: dict) -> List[tf.Tensor]:
        gradients, _ = utils.clip_gradients2(gradients, norm=self.clip_norm())
        debug['gradient_clipped_norm'] = utils.tf_norm(gradients)
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
