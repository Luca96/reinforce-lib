import gym
import matplotlib.pyplot as plt
import scipy.signal
import tensorflow_probability as tfp

from rl.agents import CategoricalReinforceAgent


def test_recent_memory():
    from rl.memories import Recent
    recent = Recent(capacity=10)

    for i in range(recent.capacity):
        recent.append(i)

    print('recent memory:')
    print(recent.buffer)
    print(recent.retrieve(5))


def test_replay_memory():
    from rl.memories import Replay
    replay = Replay(capacity=10)

    for i in range(replay.capacity):
        replay.append(i)

    print('replay memory:')
    print(replay.buffer)
    print(replay.retrieve(5))


def test_generalized_advantage_estimation(gamma: float, lambda_: float):
    rewards = [i + 1 for i in range(1, 10)]
    values = [i * i // 2 for i in range(1, 10)]
    print('rewards:', rewards)
    print('values:', values)

    def gae():
        def tf_target(t: int):
            return rewards[t] + gamma * values[t + 1] - values[t]

        advantage = 0.0
        gamma_lambda = 1

        for i in range(len(rewards) - 1):
            advantage += tf_target(i) * gamma_lambda
            gamma_lambda *= gamma * lambda_

        return advantage

    print(f'GAE({gamma}, {lambda_}) = {gae()}')


def test_gym_env(num_episodes: int, max_timesteps: int, env='CartPole-v0'):
    env = gym.make(env)
    agent = CategoricalReinforceAgent(state_shape=(4,), action_shape=(1,), batch_size=100,
                                      policy_lr=3e-3, value_lr=3e-3)

    statistics = agent.learn(env, num_episodes=num_episodes, max_timesteps=max_timesteps)

    # plot statistics
    # https://matplotlib.org/tutorials/introductory/pyplot.html
    episodes = list(range(1, num_episodes + 1))
    plt.plot(episodes, statistics['rewards'], label='reward')
    plt.plot(episodes, statistics['policy_losses'], label='policy_loss')
    plt.plot(episodes, statistics['value_losses'], label='value_loss')
    plt.legend(loc="upper left")
    plt.xlabel('episodes')
    plt.show()


def discount_cumsum(x, discount: float):
    return scipy.signal.lfilter([1.0], [1.0, float(-discount)], x[::-1], axis=0)[::-1]


def test_distribution():
    distribution = tfp.distributions.Categorical(probs=[0.7, 0.3])
    new_dist = tfp.distributions.Categorical(probs=[0.6, 0.4])
    print(distribution)

    e1 = distribution.sample(5)
    e2 = distribution.sample(5)

    print(e1, distribution.log_prob(e1), new_dist.log_prob(e1))
    print(e2, distribution.log_prob(e2))
    print(distribution.log_prob([0, 1]))
    print(distribution.log_prob([1, 0]))


def test_categorical(probs, action_shape=1):
    categorical = tfp.distributions.Categorical(probs=probs)
    # multinomial = tfp.distributions.Multinomial(probs=[0.1, 0.3, 0.6], total_count=action_shape)
    print('categorical-1:', categorical.sample(1))
    print(f'categorical-{action_shape}:', categorical.sample(sample_shape=action_shape))


def test_independent_categorical(logits: list, action_shape=None):
    ind = tfp.distributions.Independent(
        distribution=tfp.distributions.Categorical(logits=logits),
        reinterpreted_batch_ndims=1)

    print(ind.sample())


def test_normal(mean):
    normal = tfp.distributions.Normal(loc=mean, scale=mean)
    print(normal.sample(5))


def test_beta(alpha, beta):
    beta = tfp.distributions.Beta(alpha, beta)
    # samples in the range [0, 1]
    print(beta.sample(5))


if __name__ == '__main__':
    # Memories:
    # test_recent_memory()
    # test_replay_memory()

    # GAE:
    # test_generalized_advantage_estimation(gamma=0.99, lambda_=0.0)
    # test_generalized_advantage_estimation(gamma=0.99, lambda_=1.0)

    # Environments:
    # test_gym_env(num_episodes=200 * 5, max_timesteps=100, env='CartPole-v0')
    # print(discount_cumsum([1, 2, 3, 4], 1))

    # Distributions:
    # test_distribution()
    # test_categorical(action_shape=4, probs=[0.1, 0.3, 0.1, 0.5])
    # test_independent_categorical(logits=[[1, 2], [3, 4]])
    # test_normal(mean=[1.0, 2.5])
    # test_beta(alpha=[1, 1], beta=2)
    cat = tfp.distributions.Categorical(logits=[1, 2, 3, 4])
    print(cat.log_prob([[1], [2], [3]]))
    pass
