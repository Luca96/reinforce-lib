import gym

from rl import utils
from rl.agents import PPOAgent

from rl.parameters import LinearParameter
from tensorflow.keras.optimizers.schedules import ExponentialDecay


def ppo_cartpole_test():
    from tensorflow.keras.optimizers.schedules import ExponentialDecay

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
    # decaying lr improves convergence speed
    lr = ExponentialDecay(1e-3, decay_steps=2000, decay_rate=0.95, staircase=True)
    # lr = ExponentialDecay(1e-4, decay_steps=2000, decay_rate=0.95, staircase=True)
    # lr = 3e-4

    # agent = PPOAgent(env, policy_lr=lr, value_lr=lr, clip_ratio=0.05,
    #                  lambda_=0.95, entropy_regularization=0.0, name='ppo-cartPole-r',
    #                  optimization_steps=(1, 2), batch_size=20, target_kl=None,
    #                  recurrent_policy=False,
    #                  log_mode='summary', load=False, seed=42)

    # CartPole baseline agent (avg. 130 - 200)
    agent = PPOAgent(env,
                     policy_lr=ExponentialDecay(1e-3, decay_steps=2000, decay_rate=0.95, staircase=True),
                     value_lr=ExponentialDecay(1e-3, decay_steps=2000, decay_rate=0.95, staircase=True),
                     clip_ratio=0.05,
                     lambda_=0.95, entropy_regularization=0.0, name='ppo-cartPole-mixture',
                     optimization_steps=(1, 2), batch_size=20, target_kl=None,
                     recurrent_policy=False,
                     log_mode='summary', load=False, seed=42)

    # mixture-categorical baseline (smaller lr for value, more frequent decay for policy)
    # agent = PPOAgent(env,
    #                  policy_lr=ExponentialDecay(1e-3, decay_steps=2000, decay_rate=0.95, staircase=True),
    #                  value_lr=ExponentialDecay(1e-3, decay_steps=2000, decay_rate=0.95, staircase=True),
    #                  clip_ratio=0.05,
    #                  lambda_=0.95, entropy_regularization=0.0, name='ppo-cartPole-mixture',
    #                  optimization_steps=(1, 2), batch_size=20, target_kl=None,
    #                  recurrent_policy=False, mixture_components=2,
    #                  log_mode='summary', load=False, seed=42)

    agent.learn(episodes=200, timesteps=200, render_every=10, save_every='end')


def ppo_gym_test(e: int, t: int, b: int, env_name='MountainCarContinuous-v0'):
    from rl.parameters import LinearParameter
    from tensorflow.keras.optimizers.schedules import ExponentialDecay

    env = gym.make(env_name)
    utils.print_info(env)

    p_lr = ExponentialDecay(3e-4, decay_steps=2000, decay_rate=0.95, staircase=True)
    v_lr = ExponentialDecay(3e-4, decay_steps=2000, decay_rate=0.95, staircase=True)

    agent = PPOAgent(env, policy_lr=p_lr, value_lr=v_lr, clip_ratio=0.05,
                     lambda_=0.95, entropy_regularization=0.001, name=f'ppo-{env_name}',
                     optimization_steps=(1, 2), batch_size=b,
                     recurrent_policy=True,
                     load=True, log_mode='summary', seed=51)

    agent.learn(episodes=e, timesteps=t, render_every=5, save_every='end')


def ppo_pendulum(e: int, t: int, b: int, load=False):
    env = gym.make('Pendulum-v0')
    utils.print_info(env)

    p_lr = ExponentialDecay(1e-3, decay_steps=e * t, decay_rate=0.5, staircase=True)
    v_lr = ExponentialDecay(1e-3, decay_steps=e * t, decay_rate=0.5, staircase=True)

    class MyAgent(PPOAgent):
        def policy_layers(self, inputs: dict, **kwargs):
            return super().policy_layers(inputs, units=64, layers=4)

    agent = MyAgent(env, policy_lr=p_lr, value_lr=v_lr, clip_ratio=0.10,
                    entropy_regularization=0.0, name=f'ppo-Pendulum',
                    optimization_steps=(1, 2), batch_size=b, target_kl=False,
                    recurrent_policy=True, recurrent_units=16, consider_obs_every=1,
                    load=load, log_mode='summary', seed=31)

    agent.learn(episodes=e, timesteps=t, render_every=5, save_every='end')


def ppo_lunar_lander(e: int, t: int, b: int, load=False):
    env = gym.make('LunarLanderContinuous-v2')
    utils.print_info(env)

    p_lr = ExponentialDecay(1e-3 / 10, decay_steps=2000, decay_rate=0.95, staircase=True)
    v_lr = ExponentialDecay(1e-3 / 10, decay_steps=2000, decay_rate=0.95, staircase=True)

    class MyAgent(PPOAgent):
        def policy_layers(self, inputs: dict, **kwargs):
            return super().policy_layers(inputs, units=64, layers=4)

    agent = MyAgent(env, policy_lr=p_lr, value_lr=v_lr, clip_ratio=0.10 / 2,
                    entropy_regularization=0.0, name=f'ppo-LunarLander',
                    optimization_steps=(2, 2), batch_size=b, target_kl=False,
                    consider_obs_every=2,
                    recurrent_policy=True, recurrent_units=16, mixture_components=3,
                    load=load, log_mode='summary', seed=42)
    # agent.summary()
    # breakpoint()
    agent.learn(episodes=e, timesteps=t, render_every=5, save_every='end')


if __name__ == '__main__':
    # main()
    # gym_test()
    # reinforce_test()
    # ppo_cartpole_test()
    # ppo_pendulum(e=200, t=200, b=50, load=False)
    ppo_lunar_lander(e=500, t=200, b=32, load=False)
    # BipedalWalker-v2
    # ppo_gym_test(e=32, t=64, b=16, env_name='Pendulum-v0')
    pass
