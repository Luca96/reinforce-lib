
# TODO: create two directories for agents: "model_free" and "model_based" (but include all here)
from rl.v2.agents.agents import Agent, ParallelAgent, RandomAgent

# Policy optimization agents:
from rl.v2.agents.vpg import VPG
from rl.v2.agents.a2c import A2C
from rl.v2.agents.ppo import PPO
# from rl.v2.agents.a3c import A3C
# TODO: TRPO, ACER, ...

# Q-Learning agents:
from rl.v2.agents.dqn import DQN

# Policy Opt. + Q-learning agents:
from rl.v2.agents.ddpg import DDPG
from rl.v2.agents.td3 import TD3
from rl.v2.agents.sac import SAC

# Distributional RL agent:
# TODO: QR-DQN + check implementations
# from rl.v2.agents.rainbow import Rainbow
# from rl.v2.agents.iqn import IQN

# Model-based agents:
# TODO: MCTS, ...
# from rl.v2.agents.mcts import MCTS
