"""Dynamic step-dependent parameters"""

import tensorflow as tf

from typing import Union

from tensorflow.keras.optimizers import schedules
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class DynamicParameter:
    """Interface for learning rate schedule wrappers as dynamic-parameters"""
    def __init__(self):
        # self.value = 0
        # self.step = 0
        self._value = tf.Variable(initial_value=0.0, trainable=False, dtype=tf.float32)
        self.step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)

    @property
    def value(self):
        return self._value.value()

    @value.setter
    def value(self, value):
        self._value.assign(value)

    @staticmethod
    def create(value: Union[float, int, LearningRateSchedule], **kwargs):
        """Converts a floating or LearningRateSchedule `value` into a DynamicParameter object"""
        if isinstance(value, (DynamicParameter, ScheduleWrapper)):
            return value

        if isinstance(value, (float, int)):
            return ConstantParameter(value)

        if isinstance(value, LearningRateSchedule):
            return ScheduleWrapper(schedule=value, **kwargs)

        raise ValueError(f'Parameter "value" should be not {type(value)}.')

    def __call__(self, *args, **kwargs):
        return self.value

    def serialize(self) -> dict:
        # return dict(step=int(self.step))
        return dict(step=int(self.step.value()))

    def on_episode(self):
        # self.step += 1
        self.step.assign_add(delta=1)

    def load(self, config: dict):
        # self.step = config.get('step', 0)
        self.step.assign(value=config.get('step', 0))

    def get_config(self) -> dict:
        return {}


# TODO: decay on new episode (optional)
class ScheduleWrapper(LearningRateSchedule, DynamicParameter):
    """A wrapper for built-in tf.keras' learning rate schedules"""
    def __init__(self, schedule: LearningRateSchedule, min_value=1e-7):
        super().__init__()
        self.schedule = schedule
        self.min_value = tf.constant(min_value, dtype=tf.float32)

        self._value.assign(value=self.schedule.initial_learning_rate)

    def __call__(self, *args, **kwargs):
        # self.step += 1
        self.value = tf.maximum(self.min_value, self.schedule.__call__(self.step))

        return self.value

    def get_config(self) -> dict:
        return self.schedule.get_config()


# TODO: need testing (is still necessary?)
class LearnableParameter(DynamicParameter):
    def __init__(self, initial_value: float, name=None):
        self._value = tf.Variable(initial_value=initial_value, trainable=True, name=name,
                                  dtype=tf.float32)
        super().__init__()
        self.value = initial_value

    @property
    def value(self):
        return self._value.value()

    @value.setter
    def value(self, v):
        self._value.assign(value=v, read_value=False)

    @property
    def variable(self) -> list:
        return [self._value]

    def __call__(self, *args, **kwargs):
        return self.value


class ConstantParameter(DynamicParameter):
    """A constant learning rate schedule that wraps a constant float learning rate value"""
    def __init__(self, value: float):
        super().__init__()
        self.value = value
        # self._value.assign(value)

    def __call__(self, *args, **kwargs):
        return self.value

    def serialize(self) -> dict:
        return {}


class ExponentialDecay(ScheduleWrapper):
    def __init__(self, initial_value: float, decay_steps: int, decay_rate: float, staircase=False, min_value=0.0):
        super().__init__(schedule=schedules.ExponentialDecay(initial_learning_rate=initial_value,
                                                             decay_steps=decay_steps, decay_rate=decay_rate,
                                                             staircase=staircase),
                         min_value=min_value)


class StepDecay(ScheduleWrapper):
    def __init__(self, initial_value: float, decay_steps: int, decay_rate: float, min_value=1e-7):
        super().__init__(schedule=schedules.ExponentialDecay(initial_value, decay_steps, decay_rate, staircase=True),
                         min_value=min_value)


class PolynomialDecay(ScheduleWrapper):
    def __init__(self, initial_value: float, end_value: float, decay_steps: int, power=1.0, cycle=False):
        super().__init__(schedule=schedules.PolynomialDecay(initial_learning_rate=initial_value,
                                                            decay_steps=decay_steps, end_learning_rate=end_value,
                                                            power=power, cycle=cycle))
