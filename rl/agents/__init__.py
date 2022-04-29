
from rl.agents.agents import Agent, ParallelAgent, RandomAgent

# Policy optimization agents:
from rl.agents.vpg import VPG
from rl.agents.a2c import A2C
from rl.agents.ppo import PPO
# from rl.agents.a3c import A3C
# TODO: ACER, ...

# Q-Learning agents:
from rl.agents.dqn import DQN

# Policy Opt. + Q-learning agents:
from rl.agents.ddpg import DDPG
from rl.agents.td3 import TD3
from rl.agents.sac import SAC, SACDiscrete

# Distributional RL agent:
# TODO: QR-DQN + check implementations
from rl.agents.rainbow import Rainbow
from rl.agents.iqn import IQN

# Model-based agents:
# TODO: MCTS, ...
# from rl.agents.mcts import MCTS
