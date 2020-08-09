"""Learning rate schedules"""

from tensorflow.keras.optimizers import schedules


class Schedule:
    """Interface for learning rate schedule wrappers"""
    def serialize(self) -> dict:
        return {}


# TODO: decay/new lr on new episode (optional)
class ScheduleWrapper(schedules.LearningRateSchedule, Schedule):
    """A wrapper for built-in tf.keras' learning rate schedules"""
    def __init__(self, lr_schedule: schedules.LearningRateSchedule, offset=0):
        self.lr_schedule = lr_schedule
        self.lr = None

        # variables for saving/loading
        self.step_offset = offset
        self.step = 0

    def __call__(self, step):
        self.step = step + self.step_offset
        self.lr = self.lr_schedule.__call__(self.step)
        return self.lr

    def get_config(self) -> dict:
        return self.lr_schedule.get_config()

    def serialize(self) -> dict:
        return dict(step_offset=int(self.step))


class ConstantSchedule(Schedule):
    """A constant learning rate schedule that wraps a constant float learning rate value"""
    def __init__(self, lr: float):
        self.lr = lr

    def __call__(self):
        return self.lr
