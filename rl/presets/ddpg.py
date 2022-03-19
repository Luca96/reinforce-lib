
from rl.presets.preset import Preset
from rl.parameters import StepDecay


class DDPGPresets(Preset):
    # 250 episodes, 200 timesteps, exploration_steps=5 * 128
    LUNAR_LANDER_CONTINUOUS = dict(env='LunarLanderContinuous-v2', actor_lr=1e-3, critic_lr=1e-3,
                                   polyak=0.995, memory_size=256_000, batch_size=128, name='ddpg-lunar_c',
                                   actor=dict(units=64), critic=dict(units=64),
                                   noise=StepDecay(0.1, steps=50, rate=0.5), seed=42)
