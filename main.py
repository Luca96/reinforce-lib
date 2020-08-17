import gym
import numpy as np
import tensorflow as tf

from rl import utils
from rl.agents import PPOAgent

from tensorflow.keras.optimizers.schedules import ExponentialDecay


def ppo_cartpole_test():
    env = gym.make('CartPole-v0')
    utils.print_info(env)

    lr_schedule = ExponentialDecay(
        initial_learning_rate=1e-3, decay_steps=2000, decay_rate=0.95, staircase=True)

    # good seeds:  42, 31, 91
    agent = PPOAgent(env,
                     policy_lr=lr_schedule,
                     value_lr=lr_schedule,
                     clip_ratio=0.10, lambda_=0.95, entropy_regularization=0.0,
                     name='ppo-cartPole-baseline2',
                     optimization_steps=(1, 2), batch_size=20, target_kl=None,
                     accumulate_gradients_over_batches=False, polyak=0.95,
                     log_mode='summary', load=False, seed=42)

    agent.learn(episodes=200, timesteps=200, render_every=10, save_every='end')


def ppo_lunar_lander_discrete(e=200, t=200, b=20, load=False):
    env = gym.make('LunarLander-v2')
    utils.print_info(env)

    # best: (batch: 50) 500 epochs 1e-3/3e-4 + 500 epochs half lr [value_objective 2]
    agent = PPOAgent(env,
                     policy_lr=ExponentialDecay(1e-3, decay_steps=2000, decay_rate=0.95, staircase=True),
                     value_lr=ExponentialDecay(3e-4, decay_steps=2000, decay_rate=0.95, staircase=True),
                     clip_ratio=0.05 + 0.15,
                     lambda_=0.95, entropy_regularization=0.01, name='ppo-LunarLander-discrete',
                     optimization_steps=(1 + 1, 1 + 1), batch_size=b, clip_norm=(0.5, 0.5),
                     log_mode='summary', load=load, seed=42)
    agent.drop_batch_reminder = True

    # agent = PPOAgent(env,
    #                  policy_lr=ExponentialDecay(1e-3, decay_steps=2000, decay_rate=0.95, staircase=True),
    #                  value_lr=ExponentialDecay(3e-4, decay_steps=2000, decay_rate=0.95, staircase=True),
    #                  clip_ratio=0.05,
    #                  lambda_=0.95, entropy_regularization=0.001, name='ppo-LunarLander-discrete',
    #                  optimization_steps=(1, 2 - 1), batch_size=b, clip_norm=(0.5, 0.5),
    #                  network=dict(units=64, num_layers=2, activation=utils.lisht),
    #                  log_mode='summary', load=load, seed=42)
    # agent.drop_batch_reminder = True

    agent.learn(episodes=e, timesteps=t, render_every=10, save_every='end')


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

    # best: 300 episodes
    # agent = PPOAgent(env, policy_lr=1e-3, value_lr=1e-3, clip_ratio=0.15,
    #                  lambda_=0.95, entropy_regularization=0.0,
    #                  name='ppo-LunarLander', clip_norm=0.5,
    #                  optimization_steps=(1, 2), batch_size=b,
    #                  log_mode='summary', load=load, seed=123)

    agent = PPOAgent(env, policy_lr=1e-3, value_lr=1e-3, clip_ratio=0.15,
                     lambda_=0.95, entropy_regularization=0.0,
                     name='ppo-LunarLander', clip_norm=0.5,
                     optimization_steps=(1, 2), batch_size=b, polyak=0.95,
                     log_mode='summary', load=load, seed=123)

    agent.learn(episodes=e, timesteps=t, render_every=10, save_every=save_every)


def ppo_mountain_car(e: int, t: int, b: int, load=False):
    env = gym.make('MountainCar-v0')
    utils.print_info(env)
    pass


def ppo_walker(e: int, t: int, b: int, load=False, save_every='end'):
    env = gym.make('BipedalWalker-v3')
    utils.print_info(env)

    agent = PPOAgent(env, policy_lr=1e-3, value_lr=1e-3, clip_ratio=0.15,
                     lambda_=0.95, entropy_regularization=0.0,
                     name='ppo-walker', clip_norm=0.5,
                     optimization_steps=(1, 1), batch_size=b,
                     # consider_obs_every=4, accumulate_gradients_over_batches=True,
                     network=dict(units=128, num_layers=2, activation=tf.nn.relu6),
                     polyak=0.95, update_frequency=4,
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
    # ppo_cartpole_test()
    # ppo_acrobot(e=200, t=200, b=32)
    # ppo_lunar_lander_discrete(e=500, t=200, b=40, load=False)
    # ppo_pendulum(e=200, t=200, b=64, load=False)
    # ppo_lunar_lander(e=500, t=200, b=32, load=False, save_every=100)
    # ppo_mountain_car(e=400, t=1000, b=100, load=False)
    # ppo_car_racing_discrete(e=200, t=200, b=50, load=False)
    # ppo_walker(e=400, t=200, b=50, load=False)
    pass
