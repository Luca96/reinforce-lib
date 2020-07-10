
import matplotlib.pyplot as plt


from rl import parameters as param


def test_exponential_parameter(num_steps=15, **kwargs):
    p = param.ExponentialParameter(steps=num_steps, **kwargs)
    v = [p() for _ in range(num_steps)]

    plt.scatter(x=list(range(num_steps)), y=v)
    plt.title(f'({num_steps}, {kwargs["rate"]}, {round(v[-1], 2)})')
    plt.show()


def test_linear_parameter(initial=10.0, final=0.0, num_steps=15):
    p = param.LinearParameter(initial=initial, final=final, steps=num_steps)
    v = [p() for _ in range(num_steps)]

    plt.scatter(x=list(range(num_steps)), y=v)
    plt.show()


def test_rate_comparison_exp(num_steps: int, **kwargs):
    x = list(range(num_steps))

    for r in [-1, 1, 0.9, 0.5, 0.25, 0.1, 0.05, 0.001]:
        p = param.ExponentialParameter(rate=r, steps=num_steps, **kwargs)
        plt.scatter(x, y=[p() for _ in range(num_steps)])
    plt.show()


def test_rate_comparison_linear(num_steps: int, **kwargs):
    x = list(range(num_steps))

    for r in [-10, -2, -1, -0.5, -0.01, 5, 3, 2, 0.9, 0.5, 0.25, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001, 0.0]:
        p = param.LinearParameter(rate=r, steps=num_steps, **kwargs)
        v = [p() for _ in range(num_steps)]
        plt.scatter(x, v)

    plt.show()


def test_parameter_restart(which: str, num_steps: int, repeat=4, **kwargs):
    x = list(range(num_steps * repeat))
    y = []

    if which == 'exp':
        p = param.ExponentialParameter(steps=num_steps, **kwargs)
    elif which == 'step':
        p = param.StepParameter(steps=num_steps, **kwargs)
    else:
        p = param.LinearParameter(steps=num_steps, **kwargs)

    for _ in range(repeat):
        y.extend([p() for _ in range(num_steps)])

    plt.scatter(x, y)
    plt.show()


if __name__ == '__main__':
    # Exponential:
    # test_exponential_parameter(initial=2.0, final=10.0, num_steps=30)
    # test_rate_comparison_exp(num_steps=100, initial=10.0, final=2.0)

    # Linear:
    # test_linear_parameter(initial=2.5, final=10.0, num_steps=30)
    # test_rate_comparison_linear(num_steps=100, initial=10.0, final=2.5)
    # test_rate_comparison_linear((num_steps=100, final=10.0, initial=2.5)

    # Restart:
    # test_parameter_restart('linear', num_steps=50, repeat=4, initial=1.0, final=0.0, restart=True,
    #                        decay_on_restart=0.5, rate=0.9)
    # test_parameter_restart('exp', num_steps=50, repeat=4, initial=1.0, final=0.0, restart=True, rate=0.9,
    #                        decay_on_restart=0.5)
    # test_parameter_restart('step', num_steps=20, repeat=5, value=2.0, restart=True, decay_on_restart=0.75)

    # step-parameter can be used as constant parameter
    # test_parameter_restart('step', num_steps=50, value=2.0, repeat=2, restart=True)
    pass
