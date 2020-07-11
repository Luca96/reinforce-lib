
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import schedules


def test_schedule(schedule_class: schedules.LearningRateSchedule, **kwargs):
    steps = kwargs['decay_steps']
    schedule = schedule_class(**kwargs)

    x = list(range(steps * 4))
    y = [schedule(step=i) for i in x]

    plt.scatter(x, y)
    plt.show()


if __name__ == '__main__':
    # Exponential decay:
    # test_schedule(schedules.ExponentialDecay, initial_learning_rate=1.0, decay_steps=50, decay_rate=0.5)
    # test_schedule(schedules.ExponentialDecay, initial_learning_rate=1.0, decay_steps=50, decay_rate=0.5,
    #               staircase=True)

    # Inverse time decay:
    # test_schedule(schedules.InverseTimeDecay, initial_learning_rate=1.0, decay_steps=50, decay_rate=0.5)
    # test_schedule(schedules.InverseTimeDecay, initial_learning_rate=1.0, decay_steps=50, decay_rate=0.5,
    #               staircase=True)

    # Polynomial decay:
    # test_schedule(schedules.PolynomialDecay, initial_learning_rate=1.0, power=1.0, decay_steps=50, cycle=False)
    # test_schedule(schedules.PolynomialDecay, initial_learning_rate=1.0, power=0.5, decay_steps=50, cycle=True)
    # test_schedule(schedules.PolynomialDecay, initial_learning_rate=1.0, power=0.1, decay_steps=50, cycle=True)
    # test_schedule(schedules.PolynomialDecay, initial_learning_rate=1.0, power=1.5, decay_steps=50, cycle=True)
    pass
