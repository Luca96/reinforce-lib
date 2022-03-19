
import tensorflow as tf

from tensorflow import keras as tfk

from typing import Union, Optional, List


class MyLayer(tf.keras.layers.Layer):
    """Abstract base layer, meant to be used for custom layers that require additional arguments in call()"""

    def __init__(self, *args, extra_call_kwargs: Optional[List[str]] = None, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(extra_call_kwargs, list) and len(extra_call_kwargs) > 0:
            self.has_extra_call_kwargs = True
            self.extra_call_kwargs = extra_call_kwargs
            self.kwargs = {}
        else:
            self.has_extra_call_kwargs = False

    def set_kwargs(self, **kwargs):
        assert self.has_extra_call_kwargs

        for key in self.extra_call_kwargs:
            if key in kwargs:
                self.kwargs[key] = kwargs[key]


class ScaledInitializer(tfk.initializers.Initializer):
    """Wraps a tf.keras.initializers.Initializer weight-initializer instance to rescale its output"""

    def __init__(self, scaling: float, initializer: Union[str, tfk.initializers.Initializer] = 'glorot_uniform'):
        assert isinstance(scaling, float) and scaling > 0.0
        super().__init__()

        if isinstance(initializer, str):
            self.weight_init = tfk.initializers.get(identifier=initializer)
        else:
            assert isinstance(initializer, tfk.initializers.Initializer)
            self.weight_init = initializer

        self.scaling = tf.constant(scaling, dtype=tf.float32)

    def __call__(self, shape, dtype=None):
        return self.scaling * self.weight_init(shape=shape)


class Sampling(tfk.layers.Layer):
    """Given mean and log-variance that parametrize a Gaussian, the layer samples from it by using the
       reparametrization trick.
        - https://keras.io/examples/generative/vae/
    """

    def call(self, inputs, **kwargs):
        mean, log_var = inputs

        # sample from a Standard Normal
        epsilon = tf.random.normal(shape=tf.shape(mean))

        # Reparametrization trick
        return mean + tf.exp(0.5 * log_var) * epsilon


class Linear(tfk.layers.Dense):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='linear', **kwargs)
