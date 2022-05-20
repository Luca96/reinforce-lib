"""Tris (Tic-Tac-Toe) Environment"""

import gym
import sys
import numpy as np

from gym.utils import colorize as gym_colorize

from rl.environments.gym import MCTSEnv


class TrisEnv(MCTSEnv):
    """Tic-Tac-Toe textual environment"""
    CROSS = -1
    EMPTY = 0
    CIRCLE = 1

    PATTERN = ' {} │ {} │ {}\n───┼───┼───\n {} │ {} │ {}\n───┼───┼───\n {} │ {} | {}\n'

    WIN_POSITIONS = [
        # rows
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        # columns
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        # diagonals
        [0, 4, 8], [2, 4, 6]
    ]

    def __init__(self, opponent_starts=False, seed=None):
        super().__init__(seed=seed)

        self.grid = np.zeros(shape=(9,), dtype=np.int32)
        self.opponent_start_first = bool(opponent_starts)

        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int32)
        self.action_space = gym.spaces.Discrete(n=9)
        self.reward_range = (-1.0, 1.0)

    def step(self, action):
        action = np.squeeze(action).astype(np.int32)

        if not self.action_space.contains(action):
            raise RuntimeError(f'Action {action} does not belong to action space {self.action_space}.')

        available_actions = self.valid_actions()

        if action not in available_actions:
            raise RuntimeError(f'Action {action} not valid. Available actions: {available_actions}.')

        # Player move:
        self.grid[action] = self.CROSS

        if self.has_won(symbol=self.CROSS):
            # state, reward, done, info
            return self.get_state(), 1, True, {}

        elif len(available_actions) - 1 == 0:
            # no move left: draw
            return self.get_state(), 0, True, {}

        # Opponent move:
        available_actions = self.valid_actions()
        opponent_action = self.sample()
        self.grid[opponent_action] = self.CIRCLE

        if self.has_won(symbol=self.CIRCLE):
            return self.grid.copy(), -1, True, dict(opponent=opponent_action)

        elif len(available_actions) - 1 == 0:
            # no move left: draw
            return self.get_state(), 0, True, {}

        return self.get_state(), 0, False, {}

    def reset(self):
        self.grid[:] = 0  # reset to all 0s

        if self.opponent_start_first:
            self.grid[self.random.choice(9)] = self.CIRCLE

        return self.get_state()

    def render(self, mode='human'):
        # sys.stdout.write(f'\r{self.PATTERN.format(*self._flatten())}')

        # sys.stdout.write(self.PATTERN.format(*self._flatten()))
        # sys.stdout.flush()

        grid = map(self._num_to_symbol, self.grid)
        grid = self._colorize(symbols=list(grid))

        print(self.PATTERN.format(*grid))

    def close(self):
        pass

    def has_won(self, symbol: float) -> bool:
        for indices in self.WIN_POSITIONS:
            if all(self.grid[indices] == symbol):
                return True

        return False

    def get_state(self):
        return self.grid.copy()

    def valid_actions(self) -> np.ndarray:
        actions = []

        for i, cell in enumerate(self.grid):
            if cell == self.EMPTY:
                actions.append(i)

        return np.asarray(actions)

    def _num_to_symbol(self, x) -> str:
        if x == self.EMPTY:
            return ' '

        if x == self.CROSS:
            return 'X'

        return 'O'

    def _colorize(self, symbols: list) -> list:
        if self.has_won(symbol=self.CROSS):
            color = 'green'
            symbol = self.CROSS

        elif self.has_won(symbol=self.CIRCLE):
            color = 'yellow'
            symbol = self.CIRCLE
        else:
            return symbols

        # find location of winning triplet
        for indices in self.WIN_POSITIONS:
            if all(self.grid[indices] == symbol):
                for i in indices:
                    symbols[i] = gym_colorize(symbols[i], color, bold=True)

                break

        return symbols
