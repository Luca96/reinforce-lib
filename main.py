import rl
import math
import numpy as np
import gym
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from rl.agents import *
from rl.agents.reinforce import ReinforceAgent
from rl.agents.ppo import PPOAgent


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
    agent = CategoricalReinforceAgent(state_shape=(4,), action_shape=(1,), batch_size=16)
    rewards = []

    # TODO: take environments as agent's argument (when built) so that efficient training loops can be made
    # https://www.tensorflow.org/guide/function#python_or_tensor_args
    # https://keras.io/examples/rl/actor_critic_cartpole/
    for i_episode in range(200):
        observation = env.reset()
        episode_reward = 0.0

        for t in range(100):
            env.render()
            action = agent.act(observation, training=True)

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


def plot_statistics(stats: dict):
    plt.subplot(131)
    plt.plot(stats['value_losses'], color='y')
    plt.title('Value Loss')

    plt.subplot(132)
    plt.plot(stats['policy_losses'], color='b')
    plt.title('Policy Loss')

    plt.subplot(133)
    plt.plot(stats['episode_rewards'], color='g')
    plt.title('Episode Reward')

    plt.show()


def plot_statistics2(stats: dict, colormap='Pastel1'):
    num_plots = len(stats.keys())
    cmap = plt.get_cmap(name=colormap)

    if math.sqrt(num_plots) == float(math.isqrt(num_plots)):
        rows = math.isqrt(num_plots)
        cols = rows
    else:
        rows = 1
        cols = num_plots

    for k, (key, value) in enumerate(stats.items()):
        plt.subplot(rows, cols, k + 1)
        plt.plot(value, color=cmap(k + 1))
        plt.title(key)

    plt.show()


def reinforce_test(gym_env='CartPole-v0'):
    env = gym.make('CartPole-v0')
    agent = ReinforceAgent()

    agent.learn(env, episodes=400, timesteps=100)
    plot_statistics(agent.stats)


def ppo_test(gym_env='CartPole-v0'):
    env = gym.make('CartPole-v0')
    agent = PPOAgent(policy_lr=3e-3, value_lr=3e-4, clip_ratio=0.1)

    agent.learn(env, episodes=10, timesteps=100, subsampling_fraction=0.25,
                render=False, save=False)
    plot_statistics2(agent.stats)


if __name__ == '__main__':
    # main()
    # gym_test()
    # reinforce_test()
    ppo_test()
