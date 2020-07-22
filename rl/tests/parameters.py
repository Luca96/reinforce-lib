
import matplotlib.pyplot as plt


from rl import parameters as param


def test_exponential_parameter(num_steps=15, **kwargs):
    p = param.ExponentialParameter(steps=num_steps, **kwargs)
    v = [p() for _ in range(num_steps)]

    plt.scatter(x=list(range(num_steps)), y=v)
    plt.title(f'({num_steps}, {kwargs["rate"]}, {round(v[-1], 2)})')
    plt.show()


def test_linear_parameter(initial=10.0, final=0.0, rate=1.0, num_steps=15):
    p = param.LinearParameter(initial=initial, rate=rate, final=final, steps=num_steps)
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
    # test_exponential_parameter(initial=0.001, final=0.0, rate=0.99, num_steps=10_000)
    # test_rate_comparison_exp(num_steps=100, initial=10.0, final=2.0)
    # test_exponential_parameter(num_steps=100, initial=0.0001, rate=1.1, final=0.1, restart=True, decay_on_restart=1.1)

    # Linear:
    # test_linear_parameter(initial=2.5, final=10.0, num_steps=30)
    # test_rate_comparison_linear(num_steps=100, initial=10.0, final=2.5)
    # test_rate_comparison_linear((num_steps=100, final=10.0, initial=2.5)
    # test_linear_parameter(initial=1.0, final=0.01, rate=0.1, num_steps=1000)
    # test_linear_parameter(initial=0.01, final=1.0, rate=1.0, num_steps=1000)

    # Restart:
    # test_parameter_restart('linear', num_steps=50, repeat=4, initial=1.0, final=0.0, restart=True,
    #                        decay_on_restart=0.5, rate=0.9)
    # test_parameter_restart('exp', num_steps=50, repeat=4, initial=1.0, final=0.0, restart=True, rate=0.9,
    #                        decay_on_restart=0.5)
    # test_parameter_restart('step', num_steps=20, repeat=5, value=2.0, restart=True, decay_on_restart=0.75)

    # Step:
    # test_parameter_restart('step', num_steps=100, value=0.0001, repeat=20, restart=True, decay_on_restart=1.5)
    # step-parameter can be used as constant parameter
    # test_parameter_restart('step', num_steps=50, value=2.0, repeat=2, restart=True)

    # Constant Parameter
    # const = param.ConstantParameter(value=1.0)
    # plt.scatter(x=list(range(100)),
    #             y=[const() for _ in range(100)])
    # plt.show()
    pass
