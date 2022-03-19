
from rl.presets.preset import Preset


class PPOPresets(Preset):
    CART_POLE_V1 = dict(env='CartPole-v1', horizon=64, batch_size=256, optimization_epochs=10,
                        name='ppo-cart_v1', policy_lr=3e-4, num_actors=16, entropy=1e-3,
                        clip_norm=(5.0, 5.0), policy=dict(units=32), value=dict(units=64),
                        target_kl=0.3, target_mse=1.0, seed=42)
