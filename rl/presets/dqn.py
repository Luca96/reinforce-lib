
from rl.presets.preset import Preset


class DQNPresets(Preset):
    # exploration_steps = 500
    CART_POLE = dict(env='CartPole-v1', batch_size=128, policy='boltzmann', memory_size=50_000,
                     name='dqn-cartpole', clip_norm=None, lr=0.001,
                     network=dict(num_layers=2, units=64, min_max=Preset.CARTPOLE_RANGE),
                     reward_scale=1.0, prioritized=False, horizon=1, gamma=0.97,
                     polyak=1.0, update_target_network=100, double=True, dueling=False, seed=42)
