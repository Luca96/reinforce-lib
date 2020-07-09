import gym

from rl import utils
from rl.agents import PPOAgent, PPO2Agent


def ppo_cartpole_test():
    env = gym.make('CartPole-v0')
    utils.print_info(env)

    # reaches almost 100-130 as reward
    # agent = PPOAgent(env, policy_lr=3e-4, value_lr=3e-5, clip_ratio=0.20,
    #                  lambda_=0.95, entropy_regularization=0.0, name='ppo-cartPole',
    #                  optimization_steps=(1, 1),
    #                  log_mode='summary', load=True)

    # agent.learn(episodes=600, timesteps=200, batch_size=20,
    #             render_every=10, save_every='end')

    # reaches 200 as reward (also cartPole3)
    # (batch shuffling works but learning is worse)
    agent = PPOAgent(env, policy_lr=1e-3, value_lr=1e-3, clip_ratio=0.20,
                     lambda_=0.95, entropy_regularization=0.0, name='ppo-cartPole',
                     optimization_steps=(1, 1), batch_size=20, target_kl=None,
                     log_mode='summary', load=False)

    agent.learn(episodes=200, timesteps=200, render_every=10, save_every=False)


def ppo_mountaincar_test():
    env = gym.make('MountainCarContinuous-v0')
    utils.print_info(env)
    # agent = PPO2Agent(env, policy_lr=1e-3, value_lr=3e-4, clip_ratio=0.20,
    #                   lambda_=0.95, entropy_regularization=0.001, name='ppo-mountainCarContinuous',
    #                   optimization_steps=(1, 1),
    #                   advantage_weights=(2.2, 0.4),
    #                   load=False, log_mode='summary')

    agent = PPOAgent(env, policy_lr=1e-3, value_lr=3e-4, clip_ratio=0.20,
                     lambda_=0.95, entropy_regularization=0.001, name='ppo-mountainCarContinuous',
                     optimization_steps=(1, 1), batch_size=50,
                     load=False, log_mode='summary')

    agent.learn(episodes=200, timesteps=1000, render_every=5, save_every='end')


if __name__ == '__main__':
    # main()
    # gym_test()
    # reinforce_test()
    ppo_cartpole_test()
    # ppo_mountaincar_test()
