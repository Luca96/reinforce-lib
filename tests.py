

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


if __name__ == '__main__':
    # Memories:
    # test_recent_memory()
    # test_replay_memory()

    # GAE:
    # test_generalized_advantage_estimation(gamma=0.99, lambda_=0.0)
    # test_generalized_advantage_estimation(gamma=0.99, lambda_=1.0)
    pass
