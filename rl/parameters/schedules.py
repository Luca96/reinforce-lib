
"""Learning rate schedules"""

import tensorflow as tf

from rl.agents import Agent
from tensorflow.keras.optimizers import schedules


# class ScheduleWrapper:
#     """Wraps a LearningRateSchedule instance"""
#     def __init__(self, agent: Agent, schedule: schedules.LearningRateSchedule, name: str):
#         self.agent = agent
#         self.schedule = schedule
#         self.name = name
#
#     def __call__(self, step):
#         print('my __call__')
#         learning_rate = self.schedule(step)
#         self.agent.log(**{f'learning_rate_{self.name}': learning_rate})
#         return learning_rate


class Schedule:
    pass


class ExponentialSchedule(schedules.ExponentialDecay, Schedule):
    """Exponential learning rate schedule"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = None

    def __call__(self, step):
        self.lr = super().__call__(step)
        return self.lr


class PolynomialSchedule(schedules.PolynomialDecay, Schedule):
    """Polynomial learning rate schedule"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = None

    def __call__(self, step):
        self.lr = super().__call__(step)
        return self.lr


class InverseTimeSchedule(schedules.InverseTimeDecay, Schedule):
    """Inverse-time learning rate schedule"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = None

    def __call__(self, step):
        self.lr = super().__call__(step)
        return self.lr
