import gym
import numpy as np
import tensorflow as tf

from rl import utils
from rl.agents import PPOAgent

from rl.parameters import StepParameter, schedules
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras import layers


def ppo_cartpole_test():
    env = gym.make('CartPole-v0')
    utils.print_info(env)

    agent = PPOAgent(env,
                     policy_lr=schedules.ExponentialSchedule(1e-3, decay_steps=2000, decay_rate=0.95, staircase=True),
                     value_lr=schedules.ExponentialSchedule(1e-3, decay_steps=2000, decay_rate=0.95, staircase=True),
                     clip_ratio=0.05,
                     lambda_=0.95, entropy_regularization=0.0, name='ppo-cartPole-baseline',
                     optimization_steps=(1, 2), batch_size=20, target_kl=None,
                     log_mode='summary', load=False, seed=42)

    agent.learn(episodes=200, timesteps=200, render_every=10, save_every='end')


def ppo_lunar_lander_discrete(e=200, t=200, b=20, load=False):
    env = gym.make('LunarLander-v2')
    utils.print_info(env)

    # best: (batch: 50) 500 epochs 1e-3/3e-4 + 500 epochs half lr [value_objective 2]
    # agent = PPOAgent(env,
    #                  policy_lr=schedules.ExponentialSchedule(1e-3, decay_steps=2000, decay_rate=0.95, staircase=True),
    #                  value_lr=schedules.ExponentialSchedule(3e-4, decay_steps=2000, decay_rate=0.95, staircase=True),
    #                  clip_ratio=0.05,
    #                  lambda_=0.95, entropy_regularization=0.0, name='ppo-LunarLander-discrete',
    #                  optimization_steps=(1, 1), batch_size=b, clip_norm=(0.5, 0.5),
    #                  log_mode='summary', load=load, seed=42)
    # agent.drop_batch_reminder = True

    agent = PPOAgent(env,
                     policy_lr=schedules.ExponentialSchedule(1e-3, decay_steps=2000, decay_rate=0.95, staircase=True),
                     value_lr=schedules.ExponentialSchedule(3e-4, decay_steps=2000, decay_rate=0.95, staircase=True),
                     clip_ratio=0.05,
                     lambda_=0.95, entropy_regularization=0.001, name='ppo-LunarLander-discrete',
                     optimization_steps=(1, 2 - 1), batch_size=b, clip_norm=(0.5, 0.5),
                     network=dict(units=64, num_layers=2, activation=utils.lisht),
                     log_mode='summary', load=load, seed=42)
    agent.drop_batch_reminder = True

    agent.learn(episodes=e, timesteps=t, render_every=10, save_every='end')


def ppo_pendulum(e: int, t: int, b: int, load=False):
    env = gym.make('Pendulum-v0')
    utils.print_info(env)

    # p_lr = schedules.ExponentialSchedule(1e-3, decay_steps=100, decay_rate=0.95, staircase=True)
    p_lr = 1e-3

    # v_lr = schedules.ExponentialSchedule(1e-3, decay_steps=200, decay_rate=0.95, staircase=True)
    v_lr = 1e-3

    # ent = 0.01
    # ent = StepParameter(value=0.01, steps=100, decay_on_restart=0.99, restart=True)
    ent = -0.001

    agent = PPOAgent(env, policy_lr=p_lr, value_lr=v_lr,
                     clip_ratio=StepParameter(value=0.10, steps=100, decay_on_restart=0.99, restart=True),
                     entropy_regularization=ent,
                     name=f'ppo-Pendulum', optimizer='adam',
                     optimization_steps=(1, 3), batch_size=b,
                     consider_obs_every=2,
                     clip_norm=(1.0, 1.0),
                     network=dict(units=32, num_layers=3, dropout=0.0),
                     load=load, log_mode='summary', seed=123)

    agent.learn(episodes=e, timesteps=t, render_every=5, save_every='end')


def ppo_lunar_lander(e: int, t: int, b: int, load=False):
    from rl import parameters as p

    env = gym.make('LunarLanderContinuous-v2')
    utils.print_info(env)

    # p_lr = ExponentialDecay(1e-3, decay_steps=100, decay_rate=0.95, staircase=True)
    # p_lr = schedules.ExponentialSchedule(1e-3, decay_steps=100, decay_rate=0.95, staircase=True)
    p_lr = 1e-3

    # v_lr = ExponentialDecay(1e-3, decay_steps=200, decay_rate=0.95, staircase=True)
    # v_lr = schedules.ExponentialSchedule(1e-3, decay_steps=200, decay_rate=0.95, staircase=True)
    v_lr = 1e-3

    # ent = p.ExponentialParameter(initial=0.001, final=0.0, rate=0.99, steps=100_000)
    # ent = 0.1
    ent = 0.01

    agent = PPOAgent(env, policy_lr=p_lr, value_lr=v_lr, clip_ratio=0.05,
                     entropy_regularization=ent,
                     name=f'ppo-LunarLander', optimizer='adam',
                     optimization_steps=(1, 2), batch_size=b,
                     consider_obs_every=1,
                     clip_norm=(1.0, 1.0),
                     network=dict(units=64, num_layers=6, dropout=0.0),
                     load=load, log_mode='summary', seed=123)

    # agent.summary()
    # breakpoint()
    agent.learn(episodes=e, timesteps=t, render_every=5, save_every='end')


def ppo_mountain_car(e: int, t: int, b: int, load=False):
    from rl import parameters as p

    env = gym.make('MountainCar-v0')
    utils.print_info(env)

    agent = PPOAgent(env,
                     policy_lr=ExponentialDecay(1e-3, decay_steps=50, decay_rate=0.9, staircase=True),
                     value_lr=ExponentialDecay(1e-3, decay_steps=50, decay_rate=0.9, staircase=True),
                     clip_ratio=0.10,
                     entropy_regularization=0.0,
                     name=f'ppo-MountainCar',
                     optimization_steps=(1, 2), batch_size=b, target_kl=False,
                     consider_obs_every=4, clip_norm=(1.0, 2.0),
                     network=dict(units=32, num_layers=6, dropout=0.0),
                     load=load, log_mode='summary', seed=42)

    agent.learn(episodes=e, timesteps=t, render_every=5, save_every='end')


def ppo_car_racing_discrete(e: int, t: int, b: int, load=False):
    from rl.environments.gym import CarRacingDiscrete
    from rl.networks import PPONetwork, shufflenet_v2

    class MyNetwork(PPONetwork):
        def policy_layers(self, inputs: dict, last_units=128, **kwargs):
            return shufflenet_v2(inputs['state'], leak=0.0, linear_units=last_units, **kwargs)

    agent = PPOAgent(CarRacingDiscrete(bins=8),
                     policy_lr=schedules.ExponentialSchedule(1e-3, decay_steps=100, decay_rate=0.95, staircase=True),
                     value_lr=schedules.ExponentialSchedule(1e-3, decay_steps=200, decay_rate=0.95, staircase=True),
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
    # main()
    # gym_test()
    # reinforce_test()
    # ppo_cartpole_test()
    ppo_lunar_lander_discrete(e=500, t=200, b=40, load=False)
    # ppo_pendulum(e=200, t=200, b=64, load=False)
    # ppo_lunar_lander(e=500, t=200, b=32, load=False)
    # ppo_mountain_car(e=400, t=1000, b=100, load=False)

    # ppo_car_racing_discrete(e=200, t=200, b=50, load=False)
    pass
