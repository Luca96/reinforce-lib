import rl
import numpy as np
import gym
import tensorflow_probability as tfp


from rl.agents import *


def gaussian_policy(action_shape):
    pass


def main():
    n = tfp.distributions.Normal(loc=[1.0, -1.0], scale=[1.0, 1.0])
    print(n.sample(1, seed=42).numpy())
    # print(n.log_prob(v).numpy())
    # print(n.prob(v).numpy())

    mvn = tfp.distributions.MultivariateNormalDiag(loc=[1., -1], scale_diag=[1.0, 1.0])
    print(mvn.sample(1, seed=42).numpy())


def gym_test():
    env = gym.make('CartPole-v0')
    agent = CategoricalReinforceAgent(state_shape=(4,), action_shape=(1,), batch_size=16,
                                      learning_rate=0.001)
    rewards = []

    # TODO: take environments as agent's argument (when built) so that efficient training loops can be made
    # https://www.tensorflow.org/guide/function#python_or_tensor_args
    # https://keras.io/examples/rl/actor_critic_cartpole/
    for i_episode in range(200):
        observation = env.reset()
        episode_reward = 0.0

        for t in range(100):
            env.render()
            action = agent.act(observation)

            observation, reward, done, info = env.step(action)
            episode_reward += reward

            agent.observe(next_state=observation, reward=reward, terminal=done)

            if done:
                print(f"Episode finished after {(t + 1)} timesteps with reward {round(episode_reward, 2)}")
                rewards.append(episode_reward)
                break

    env.close()

    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.plot(agent.statistics['losses'])
    plt.show()

    agent.close()


if __name__ == '__main__':
    # main()
    gym_test()
