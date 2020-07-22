import gym

from rl import utils
from rl.agents import PPOAgent

from rl.parameters import LinearParameter, schedules
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


def ppo_pendulum(e: int, t: int, b: int, load=False):
    env = gym.make('Pendulum-v0')
    utils.print_info(env)

    class MyAgent(PPOAgent):
        def policy_layers(self, inputs: dict, units=64, activation='swish', num_layers=6, dropout_rate=0.0):
            x = layers.Dense(units, activation='tanh')(inputs['state'])
            x = layers.BatchNormalization()(x)

            for _ in range(0, num_layers, 2):
                x = layers.Dense(units, activation=activation)(x)
                x = layers.Dropout(rate=dropout_rate)(x)

                x = layers.Dense(units, activation=activation)(x)
                x = layers.Dropout(rate=dropout_rate)(x)
                x = layers.BatchNormalization()(x)

            return x

    agent = MyAgent(env,
                    policy_lr=ExponentialDecay(1e-3, decay_steps=50, decay_rate=0.9, staircase=True),
                    value_lr=ExponentialDecay(1e-3, decay_steps=50, decay_rate=0.9, staircase=True),
                    clip_ratio=0.10,
                    entropy_regularization=0.0,
                    name=f'ppo-Pendulum',
                    optimization_steps=(1, 2), batch_size=b, target_kl=False,
                    consider_obs_every=4, clip_norm=None,
                    recurrence=dict(units=16, depth=0),
                    load=load, log_mode='summary', seed=42)

    agent.learn(episodes=e, timesteps=t, render_every=5, save_every='end')


def ppo_lunar_lander(e: int, t: int, b: int, load=False):
    from rl import parameters as p

    env = gym.make('LunarLanderContinuous-v2')
    utils.print_info(env)

    p_lr = ExponentialDecay(1e-3 / 10, decay_steps=2000, decay_rate=0.95, staircase=True)
    v_lr = ExponentialDecay(1e-3 / 10, decay_steps=2000, decay_rate=0.95, staircase=True)

    class MyAgent(PPOAgent):
        def policy_layers(self, inputs: dict, **kwargs):
            return super().policy_layers(inputs, units=64, activation='relu', layers=8)

    #
    # agent = MyAgent(env, policy_lr=p_lr, value_lr=v_lr, clip_ratio=0.10 / 2,
    #                 entropy_regularization=0.0, name=f'ppo-LunarLander',
    #                 optimization_steps=(2, 2), batch_size=b, target_kl=False,
    #                 consider_obs_every=2,
    #                 recurrent_policy=True, recurrent_units=16, mixture_components=3,
    #                 load=load, log_mode='summary', seed=42)

    # v2
    class MyAgent2(PPOAgent):
        def policy_layers(self, inputs: dict, units=64, activation='swish', num_layers=6, dropout_rate=0.0):
            x = layers.Dense(units, activation='tanh')(inputs['state'])
            x = layers.LayerNormalization()(x)

            for _ in range(0, num_layers, 2):
                x = layers.Dense(units, activation=activation)(x)
                x = layers.Dropout(rate=dropout_rate)(x)

                x = layers.Dense(units, activation=activation)(x)
                x = layers.Dropout(rate=dropout_rate)(x)
                x = layers.LayerNormalization()(x)

            return x

        def value_layers(self, inputs: dict, **kwargs):
            return self.policy_layers(inputs, activation='tanh', units=80, dropout_rate=0.0)

        # def policy_layers(self, inputs: dict, **kwargs) -> layers.Layer:
        #     return super().policy_layers(inputs, units=64, layers=8, dropout=0.0)

    # agent = MyAgent2(env,
    #                  # policy_lr=ExponentialDecay(1e-3 / 10, decay_steps=50, decay_rate=0.5, staircase=True),
    #                  policy_lr=1e-3,
    #                  value_lr=3e-4,
    #                  # value_lr=ExponentialDecay(1e-3 / 10, decay_steps=50, decay_rate=0.5, staircase=True),
    #                  clip_ratio=0.10,
    #                  # entropy_regularization=p.StepParameter(value=0.001, steps=5 * b, restart=True,
    #                  #                                        decay_on_restart=1.025),
    #                  entropy_regularization=0.0,
    #                  name=f'ppo-LunarLander', optimizer='adam',
    #                  optimization_steps=(1, 2), batch_size=b, target_kl=False,
    #                  consider_obs_every=1,  clip_norm=(1.0, 1.0),
    #                  recurrence=dict(units=16, depth=0),
    #                  load=load, log_mode='summary', seed=123)

    # ---------------------------------------------------------------------------------

    class MyAgent3(PPOAgent):
        def policy_layers(self, inputs: dict, units=64, num_layers=6, dropout=0.2):
            return super().policy_layers(inputs, units=units, layers=num_layers, dropout=dropout)

        def value_layers(self, inputs: dict, **kwargs):
            return self.policy_layers(inputs)

    # p_lr = ExponentialDecay(1e-3, decay_steps=100, decay_rate=0.95, staircase=True)
    # p_lr = schedules.ExponentialSchedule(1e-3, decay_steps=100, decay_rate=0.95, staircase=True)
    p_lr = 1e-3

    # v_lr = ExponentialDecay(1e-3, decay_steps=200, decay_rate=0.95, staircase=True)
    # v_lr = schedules.ExponentialSchedule(1e-3, decay_steps=200, decay_rate=0.95, staircase=True)
    v_lr = 1e-3

    # ent = p.ExponentialParameter(initial=0.001, final=0.0, rate=0.99, steps=100_000)
    # ent = 0.1
    ent = 0.01

    agent = MyAgent3(env, policy_lr=p_lr, value_lr=v_lr, clip_ratio=0.05,
                     entropy_regularization=ent,
                     name=f'ppo-LunarLander', optimizer='adam',
                     optimization_steps=(1, 2), batch_size=b,
                     consider_obs_every=1,
                     clip_norm=(1.0, 1.0),
                     # recurrence=dict(units=16, depth=0),
                     load=load, log_mode='summary', seed=123)

    # agent.summary()
    # breakpoint()
    agent.learn(episodes=e, timesteps=t, render_every=5, save_every='end')


def ppo_mountain_car(e: int, t: int, b: int, load=False):
    from rl import parameters as p

    env = gym.make('MountainCar-v0')
    utils.print_info(env)

    class MyAgent(PPOAgent):
        def policy_layers(self, inputs: dict, units=32, activation='swish', num_layers=6, dropout_rate=0.0):
            x = layers.Dense(units, activation='tanh')(inputs['state'])
            x = layers.LayerNormalization()(x)

            for _ in range(0, num_layers, 2):
                x = layers.Dense(units, activation=activation)(x)
                x = layers.Dropout(rate=dropout_rate)(x)

                x = layers.Dense(units, activation=activation)(x)
                x = layers.Dropout(rate=dropout_rate)(x)
                x = layers.LayerNormalization()(x)

            return x

    agent = MyAgent(env,
                    policy_lr=ExponentialDecay(1e-3, decay_steps=50, decay_rate=0.9, staircase=True),
                    value_lr=ExponentialDecay(1e-3, decay_steps=50, decay_rate=0.9, staircase=True),
                    clip_ratio=0.10,
                    entropy_regularization=0.0,
                    name=f'ppo-MountainCar',
                    optimization_steps=(1, 2), batch_size=b, target_kl=False,
                    consider_obs_every=4, clip_norm=(1.0, 2.0),
                    recurrence=dict(units=16, depth=0),
                    load=load, log_mode='summary', seed=42)

    agent.learn(episodes=e, timesteps=t, render_every=5, save_every='end')


if __name__ == '__main__':
    # main()
    # gym_test()
    # reinforce_test()
    # ppo_cartpole_test()
    # ppo_pendulum(e=200, t=200, b=50, load=False)
    ppo_lunar_lander(e=500, t=200, b=32, load=False)
    # ppo_mountain_car(e=400, t=1000, b=100, load=False)
    pass
