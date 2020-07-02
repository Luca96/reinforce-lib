import gym

from rl import utils
from rl.agents import PPOAgent


def ppo_cartpole_test():
    env = gym.make('CartPole-v0')
    utils.print_info(env)

    # agent = PPOAgent(env, policy_lr=1e-3, value_lr=1e-3, clip_ratio=0.20,
    #                  lambda_=0.95, entropy_regularization=0.0, name='ppo-cartPole',
    #                  optimization_steps=(10, 10),
    #                  use_log=True, use_summary=True)

    agent = PPOAgent(env, policy_lr=3e-4, value_lr=3e-4, clip_ratio=0.20,
                     lambda_=0.95, entropy_regularization=0.0, name='ppo-cartPole',
                     optimization_steps=(10, 10),
                     exploration='rnd', advantage_weights=(1.0, 0.5),
                     use_log=True, use_summary=True, load=False)

    agent.learn(episodes=200, timesteps=200, batch_size=20,
                render_every=10, save_every=False)


def ppo_mountaincar_test():
    env = gym.make('MountainCarContinuous-v0')
    utils.print_info(env)
    agent = PPOAgent(env, policy_lr=3e-4, value_lr=1e-4, clip_ratio=0.20,
                     lambda_=0.95, entropy_regularization=0.001, name='ppo-mountainCarContinuous',
                     optimization_steps=(10, 10),
                     exploration='rnd', advantage_weights=(2.2, 0.4),
                     use_log=True, load=False, use_summary=True)

    agent.learn(episodes=200, timesteps=1000, batch_size=32,
                render_every=5, save_every=False)


if __name__ == '__main__':
    # main()
    # gym_test()
    # reinforce_test()
    ppo_cartpole_test()
    # ppo_mountaincar_test()
