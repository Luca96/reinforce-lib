
# TODO: create two directories for agents: "model_free" and "model_based" (but include all here)
from rl.v2.agents.agents import Agent, ParallelAgent, RandomAgent

# Policy optimization agents:
from rl.v2.agents.ppo import PPO1, PPO2
from rl.v2.agents.a2c import A2C
from rl.v2.agents.a3c import A3C
# TODO: TRPO, ACER, ...

# Q-Learning agents:
from rl.v2.agents.dqn import DQN

# Policy Opt. + Q-learning agents:
from rl.v2.agents.sac import SAC
from rl.v2.agents.ddpg import DDPG
# TODO: TD3?

# Distributional RL agent:
# TODO: QR-DQN
from rl.v2.agents.rainbow import Rainbow
from rl.v2.agents.iqn import IQN

# Model-based agents:
# TODO: MCTS, ...
# from rl.v2.agents.mcts import MCTS
