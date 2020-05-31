import gym
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    # Memories:
    # test_recent_memory()
    # test_replay_memory()

    # GAE:
    # test_generalized_advantage_estimation(gamma=0.99, lambda_=0.0)
    # test_generalized_advantage_estimation(gamma=0.99, lambda_=1.0)

    # Environments:
    test_gym_env(num_episodes=200 * 5, max_timesteps=100, env='CartPole-v0')
    pass
