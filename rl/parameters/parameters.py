"""Dynamic step-dependent parameters"""

from tensorflow.keras.optimizers.schedules import LearningRateSchedule, ExponentialDecay


class DynamicParameter:
    """Interface for learning rate schedule wrappers as dynamic-parameters"""
    def __init__(self):
        self.value = 0
        self.step = 0

    def serialize(self) -> dict:
        return dict(step=int(self.step))

    def load(self, config: dict):
        self.step = config.get('step', 0)


# TODO: decay on new episode (optional)
class ParameterWrapper(LearningRateSchedule, DynamicParameter):
    """A wrapper for built-in tf.keras' learning rate schedules"""
    def __init__(self, schedule: LearningRateSchedule):
        super().__init__()
        self.schedule = schedule

    def __call__(self, *args, **kwargs):
        self.step += 1
        self.value = self.schedule.__call__(self.step)
        return self.value

    def get_config(self) -> dict:
        return self.schedule.get_config()


class ConstantParameter(DynamicParameter):
    """A constant learning rate schedule that wraps a constant float learning rate value"""
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def __call__(self, *args, **kwargs):
        return self.value

    def serialize(self) -> dict:
        return {}


class StepDecay(ParameterWrapper):
    def __init__(self, initial_learning_rate: float, decay_steps: int, decay_rate: float):
        super().__init__(schedule=ExponentialDecay(initial_learning_rate, decay_steps, decay_rate, staircase=True))
