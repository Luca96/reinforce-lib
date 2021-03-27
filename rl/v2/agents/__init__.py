
from rl.v2.agents.agents import Agent, ParallelAgent

# Policy optimization agents:
from rl.v2.agents.ppo import PPO1, PPO2
from rl.v2.agents.a2c import A2C
from rl.v2.agents.a3c import A3C
# TODO: TRPO, ACER, ...

# Q-Learning agents:
from rl.v2.agents.dqn import DQN
# TODO: Rainbow

# Policy Opt. + Q-learning agents:
from rl.v2.agents.sac import SAC
from rl.v2.agents.ddpg import DDPG
# TODO: TD3?

# Distributional RL agent:
# TODO: C51, QR-DQN, DQN-IQN

# Model-based agents:
# TODO: MCTS, ...
