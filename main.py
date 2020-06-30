import rl
import math
import numpy as np
import gym
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from rl.agents import utils
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


def reinforce_test(gym_env='CartPole-v0'):
    env = gym.make('CartPole-v0')
    agent = ReinforceAgent()

    agent.learn(env, episodes=400, timesteps=100)
    plot_statistics(agent.stats)


def ppo_cartpole_test():
    env = gym.make('CartPole-v0')
    utils.print_info(env)
    agent = PPOAgent(env, policy_lr=1e-3, value_lr=1e-3, clip_ratio=0.20,
                     lambda_=0.95, entropy_regularization=0.0, name='ppo-cartPole',
                     optimization_steps=(10, 10), early_stop=False,
                     use_log=True, use_summary=True)

    agent.learn(episodes=200, timesteps=200, batch_size=20,
                render_every=5, save_every=-1)

    # agent.plot_statistics()


def ppo_mountaincar_test():
    env = gym.make('MountainCarContinuous-v0')
    utils.print_info(env)
    agent = PPOAgent(env, policy_lr=3e-4, value_lr=1e-4, clip_ratio=0.20,
                     lambda_=0.95, entropy_regularization=0.1, name='ppo-mountainCarContinuous',
                     optimization_steps=(10, 10), early_stop=False,
                     use_log=True, load=False, use_summary=True)

    agent.learn(episodes=200, timesteps=200, batch_size=32,
                render_every=5, save_every=10)


if __name__ == '__main__':
    # main()
    # gym_test()
    # reinforce_test()
    # ppo_cartpole_test()
    ppo_mountaincar_test()
