import gym
import numpy as np
import tensorflow as tf

from rl import utils
from rl.agents import *
from rl.parameters import StepDecay, ExponentialDecay


def ppo_cartpole_test(b=20, seed=42, seeds=(42, 31, 91)):
    env = gym.make('CartPole-v0')
    utils.print_info(env)

    for seed in seeds:
        # TODO: test with lambda=1
        agent = PPOAgent(env, name='ppo-cartPole-baseline2',
                         # policy_lr=1e-3 / 8,
                         policy_lr=StepDecay(1e-3, decay_rate=0.5, decay_steps=80, min_value=1e-7),
                         value_lr=3e-4, clip_ratio=0.20,
                         optimization_steps=(1, 2), batch_size=b,
                         shuffle=True,
                         log_mode='summary', load=False, seed=seed)

        agent.learn(episodes=200, timesteps=200, render_every=10, save_every='end')


def cartpole_test(e: int, t: int):
    agent = PPOAgent(env='CartPole-v0', name='ppo-cartPole-baseline', batch_size=20,
                     log_mode='summary', load=False, seed=42, traces_dir='traces')

    # agent.evaluate(episodes=e, timesteps=t)
    # agent.collect(episodes=e, timesteps=t, record_threshold=200.0)
    agent.imitate(epochs=1)
    agent.evaluate(episodes=50, timesteps=t)


def ppo_lunar_lander_discrete(e=200, t=200, b=20, load=False, evaluate=False):
    env = gym.make('LunarLander-v2')
    utils.print_info(env)

    agent = PPOAgent(env,
                     policy_lr=ExponentialDecay(1e-3, decay_steps=2000, decay_rate=0.95, staircase=True),
                     value_lr=ExponentialDecay(3e-4, decay_steps=2000, decay_rate=0.95, staircase=True),
                     clip_ratio=0.20, entropy_regularization=0.01, name='ppo-LunarLander-discrete',
                     optimization_steps=(2, 2), batch_size=b, clip_norm=(0.5, 0.5),
                     network=dict(value=dict(components=3)),
                     log_mode=None, load=load, seed=42)

    if not evaluate:
        agent.learn(episodes=e, timesteps=t, render_every=10, save_every='end')
    else:
        agent.evaluate(episodes=100, timesteps=t)


def ppo_pendulum(e: int, t: int, b: int, load=False):
    env = gym.make('Pendulum-v0')
    utils.print_info(env)
    pass


def ppo_acrobot(e: int, t: int, b: int, load=False):
    env = gym.make('Acrobot-v1')
    utils.print_info(env)

    agent = PPOAgent(env, policy_lr=1e-3, value_lr=1e-3,
                     clip_ratio=0.10, lambda_=0.95, entropy_regularization=0.0,
                     name='ppo-acrobot',
                     optimization_steps=(1, 2), batch_size=b, target_kl=None,
                     log_mode='summary', load=load, seed=123)

    agent.learn(episodes=e, timesteps=t, render_every=10, save_every='end')


def ppo_lunar_lander(e: int, t: int, b: int, load=False, save_every='end'):
    env = gym.make('LunarLanderContinuous-v2')
    utils.print_info(env)

    agent = PPOAgent(env, name='ppo-LunarLander',
                     # decay at 100~120 (original: 80 * 3)
                     # also decay value (every 200~250 or more?)
                     policy_lr=StepDecay(1e-3, decay_rate=0.5, decay_steps=85, min_value=1e-7),
                     value_lr=3e-4, clip_ratio=0.20, lambda_=0.99,
                     optimization_steps=(1 + 1, 2 - 1), batch_size=b,
                     shuffle=True, polyak=0.999,
                     log_mode='summary', load=load, seed=42)

    # agent.summary()
    # breakpoint()

    agent.learn(episodes=e, timesteps=t, render_every=10, save_every=save_every)


def ppo_mountain_car(e: int, t: int, b: int, load=False):
    env = gym.make('MountainCar-v0')
    utils.print_info(env)
    pass


def ppo_walker(e: int, t: int, b: int, load=False, save_every='end'):
    env = gym.make('BipedalWalker-v3')
    utils.print_info(env)

    # agent = PPOAgent(env, policy_lr=1e-3, value_lr=1e-3, clip_ratio=0.15,
    #                  lambda_=0.95, entropy_regularization=0.0,
    #                  name='ppo-walker', clip_norm=0.5,
    #                  optimization_steps=(1, 1), batch_size=b,
    #                  # consider_obs_every=4, accumulate_gradients_over_batches=True,
    #                  network=dict(units=128, num_layers=2, activation=tf.nn.relu6),
    #                  polyak=0.95, update_frequency=4,
    #                  log_mode='summary', load=load, seed=42)

    agent = PPOAgent(env, name='ppo-walker',
                     # policy_lr=StepDecay(1e-3, decay_rate=0.5, decay_steps=100, min_value=1e-7),
                     policy_lr=1e-4,
                     # value_lr=3e-4,
                     value_lr=1e-4,
                     clip_ratio=0.20 - 0.05, lambda_=0.99,
                     # optimization_steps=(2 + 1, 1 + 1), batch_size=b,
                     optimization_steps=(2 + 1 + 2, 1), batch_size=b,
                     shuffle=True, polyak=0.999,
                     clip_norm=0.5,
                     entropy_regularization=0.1,
                     log_mode='summary', load=load, seed=42)

    agent.learn(episodes=e, timesteps=t, render_every=10, save_every=save_every)


def ppo_car_racing_discrete(e: int, t: int, b: int, load=False):
    from rl.environments.gym import CarRacingDiscrete
    from rl.networks import PPONetwork, shufflenet_v2

    class MyNetwork(PPONetwork):
        def policy_layers(self, inputs: dict, last_units=128, **kwargs):
            return shufflenet_v2(inputs['state'], leak=0.0, linear_units=last_units, **kwargs)

    agent = PPOAgent(CarRacingDiscrete(bins=8),
                     policy_lr=ExponentialDecay(1e-3, decay_steps=100, decay_rate=0.95, staircase=True),
                     value_lr=ExponentialDecay(1e-3, decay_steps=200, decay_rate=0.95, staircase=True),
                     clip_ratio=0.10,
                     entropy_regularization=-1.0,
                     name='ppo-CarRacing-discrete',
                     consider_obs_every=4,
                     optimization_steps=(1, 1), batch_size=b,
                     network=dict(network=MyNetwork, g=0.5, last_channels=512),
                     log_mode='summary', load=load, seed=42)
    # agent.summary()
    # breakpoint()
    agent.learn(episodes=e, timesteps=t, save_every=100, render_every=5)


if __name__ == '__main__':
    # ppo_cartpole_test(b=200, seed=None)
    # ppo_acrobot(e=200, t=200, b=32)
    # ppo_lunar_lander_discrete(e=500, t=200, b=50, load=True)
    # ppo_pendulum(e=200, t=200, b=64, load=False)
    # ppo_lunar_lander(e=500, t=200, b=32, load=True, save_every=100)
    # ppo_mountain_car(e=400, t=1000, b=100, load=False)
    # ppo_car_racing_discrete(e=200, t=200, b=50, load=False)
    # ppo_walker(e=500, t=200, b=32, load=True)

    # DQNAgent.test(env='CartPole-v0', batch_size=64, memory_size=2048, lr=3e-4, seed=42,
    #               epsilon=0.1, clip_norm=None,
    #               summary_keys=['episode_rewards', 'loss', 'q_values', 'targets'],
    #               args=dict(episodes=208, timesteps=200, render_every=10, save_every='end'))

    # DuelingDQNAgent.test(env='CartPole-v0', batch_size=32, memory_size=2048, lr=3e-4, seed=42,
    #                      epsilon=0.1,
    #                      summary_keys=['episode_rewards', 'loss', 'q_values', 'targets'],
    #                      args=dict(episodes=208, timesteps=200, render_every=10, save_every='end'))

    from rl.agents.mba import MBAAgent

    MBAAgent.test(env='CartPole-v0', name='mba-cartpole', action_lr=3e-5, network=dict(units=32),
                  batch_size=32, lr=3e-4, noise=0.0, planning_horizon=16, network_summary=False,
                  args=dict(episodes=200, timesteps=200, render_every=10, save_every='end'))
    exit()

    from rl.agents.mpc import MPCAgent

    def cartpole_reward(env, s, a):
        x = s[0]
        theta = s[2]
        done = bool(x < -env.x_threshold
            or x > env.x_threshold
            or theta < -env.theta_threshold_radians
            or theta > env.theta_threshold_radians)

        return 0.0 if done else 1.0

    MPCAgent.test(env='CartPole-v0', reward_fn=lambda s, a: 1, name='mpc-cartpole',
                  batch_size=32, lr=1e-3, noise=0.0, planning_horizon=16,
                  plan_trajectories=64, optimization_steps=8,
                  args=dict(episodes=200, timesteps=200, render_every=10, save_every='end'))
    exit()

    actor = dict(units=32, activation=tf.nn.swish,
                 kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.5),
                 bias_initializer=tf.keras.initializers.RandomUniform(-0.05, 0.05))

    critic = dict(units=32, activation=tf.nn.swish,
                  kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                  bias_initializer=tf.keras.initializers.RandomUniform(-0.05, 0.05))
    import random
    seed = random.randint(1, 1000)
    print(seed)
    DDPGAgent.test(env='CartPole-v0', batch_size=40, memory_size=4096, seed=seed,
                   actor_lr=1e-3, critic_lr=1e-3, optimization_steps=2,
                   critic=critic, actor=actor, clip_norm=1.0,
                   noise=0.05, name='ddpg-cartpole',
                   # summary_keys=['episode_rewards', 'loss', 'q_values', 'targets'],
                   args=dict(episodes=200, timesteps=200, render_every=10, save_every='end'))
    exit()

    # SACAgent.test(env='LunarLanderContinuous-v2', batch_size=128, memory_size=4096, seed=31,
    #               name='sac-lunar', temperature=0.2, lr=1e-4,
    #               args=dict(episodes=500, timesteps=200, render_every=10, save_every='end'),
    #               network_summary=False)

    # from rl.agents.auto_sac import AutoSACAgent
    # AutoSACAgent.test(env='CartPole-v0', batch_size=16, memory_size=4096, seed=42,
    #                   name='auto-sac-cartpole',
    #                   args=dict(episodes=500, timesteps=200, render_every=10, save_every='end'),
    #                   network_summary=False)
    pass
