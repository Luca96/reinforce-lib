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

    # cartpole_test(e=300, t=200)
    # ppo_cartpole_test(seeds=[42])

    # import matplotlib.pyplot as plt
    # # data = tf.random.uniform((40,), minval=-2.0, maxval=5.0)
    #
    # data = tf.random.uniform((40,), minval=-0.2, maxval=2.0)
    # # data = tf.random.uniform((40,), minval=-2.0, maxval=0.1)
    #
    # data_norm = utils.tf_normalize(data)
    # data_spn = utils.tf_sign_preserving_normalization(data)
    # data_sp2 = utils.tf_sp_norm(data)
    #
    # def scatter(aa, c: str):
    #     pos = tf.where(aa < 0.01, aa, 0.0)
    #     idx = tf.constant(list(range(40)), tf.float32)
    #     a = list(filter(lambda x: x[0] != 0.0, zip(pos, idx)))
    #     z = list(map(lambda x: x[0], a))
    #     y = list(map(lambda x: x[1], a))
    #     plt.scatter(y, z, c=c)
    #
    # scatter(data_sp2, c='violet')
    # scatter(data, c='b')
    #
    # print('mean:', tf.reduce_mean(data), 'std:', tf.math.reduce_std(data))
    # print('min:', tf.reduce_min(data), 'max:', tf.reduce_max(data))
    #
    # for a, b, c in zip(data, data_norm, data_spn):
    #     print(f'{np.round(a, 4)} -> {np.round(b, 4)} vs {np.round(c, 4)}')
    #
    # mean = tf.reduce_mean(data)
    # # std = tf.math.reduce_std(data)
    # ones = tf.ones_like(data)
    # #
    # plt.plot(data, label='A(s,a)')
    # # # plt.plot(data * 0.5, label='data-0.5')
    # # plt.plot(mean * ones, label='mean')
    # plt.plot(ones, label='+1')
    # plt.plot(ones * 0.0, label='zero')
    # plt.plot(-ones, label='-1')
    # # plt.plot(std * ones, label='std')
    # # plt.plot(-std * ones, label='-std')
    # # plt.plot((mean + std) * ones, label='mean+std')
    # # plt.plot((mean - std) * ones, label='mean-std')
    # # plt.plot(data_norm, label='A-norm')
    # # plt.plot(data_spn, label='A-sp')
    # plt.plot(data_sp2, label='A-sp')
    # # plt.plot(data / std, label='data / std')
    # # plt.plot(data / tf.norm(data), label='data / norm')
    # plt.legend(loc='best')
    # plt.title('Advantage Normalization')
    # plt.show()

    # from rl.agents import ReinforceAgent
    # agent = ReinforceAgent(env='CartPole-v0', name='reinforce-cartPole', batch_size=20, drop_batch_remainder=True,
    #                        lambda_=0.95, optimization_steps=(1, 1),
    #                        seed=42, episodes_per_update=1, log_mode='summary')
    # agent.learn(episodes=200, timesteps=200, render_every=10, save_every='end')

    # agent = PPOAgent(env='CartPole-v0', name='ppo2-cartpole', batch_size=32,
    #                  drop_batch_remainder=True, optimization_steps=(1, 1),
    #                  summary_keys=['episode_rewards', 'loss_policy', 'ratio'],
    #                  policy_lr=3e-4, entropy_regularization=0.0, update_frequency=4, seed=42)
    # agent.learn(episodes=192+16, timesteps=200, render_every=10, save_every=False)

    # DQNAgent.test(env='CartPole-v0', batch_size=32, memory_size=2048, lr=3e-4, seed=42,
    #               epsilon=0.1,
    #               summary_keys=['episode_rewards', 'loss', 'q_values', 'targets'],
    #               args=dict(episodes=208, timesteps=200, render_every=10, save_every='end'))

    DuelingDQNAgent.test(env='CartPole-v0', batch_size=32, memory_size=2048, lr=3e-4, seed=42,
                         epsilon=0.1,
                         summary_keys=['episode_rewards', 'loss', 'q_values', 'targets'],
                         args=dict(episodes=208, timesteps=200, render_every=10, save_every='end'))
    pass
