
from rl.presets.preset import Preset
from rl.layers.preprocessing import MinMaxScaling


class DQNPresets(Preset):
    # exploration_steps = 500
    CART_POLE = dict(env='CartPole-v1', batch_size=128, policy='boltzmann', memory_size=50_000,
                     name='dqn-cart_v1', clip_norm=None, lr=0.001,
                     network=dict(num_layers=2, units=64,
                                  preprocess=dict(state=MinMaxScaling(min_value=Preset.CARTPOLE_MIN,
                                                                      max_value=Preset.CARTPOLE_MAX))),
                     reward_scale=1.0, prioritized=False, horizon=1, gamma=0.97,
                     polyak=1.0, update_target_network=100, double=True, dueling=False, seed=42)

    # learns perfect balance in just 50 episodes
    CART_POLE2 = dict(env='CartPole-v1', batch_size=128, policy='e-greedy', memory_size=50_000,
                      name='dqn-cart_v1', clip_norm=None, lr=0.001, epsilon=0.2,
                      network=dict(num_layers=2, units=64,
                                   preprocess=dict(state=MinMaxScaling(min_value=Preset.CARTPOLE_MIN,
                                                                       max_value=Preset.CARTPOLE_MAX))),
                      reward_scale=1.0, prioritized=False, horizon=1, gamma=0.97,
                      polyak=1.0, update_target_network=100, double=True, dueling=False, seed=42)
