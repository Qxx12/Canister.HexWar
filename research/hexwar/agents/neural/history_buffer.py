"""
Per-game history buffer for frame-stacked temporal encoding.

Stores the last K board snapshots so the encoder can build the full
temporal feature tensor. The buffer is maintained by the PPOAgent or
HexWarEnv and passed to encode_board_with_history().

Why frame stacking over LSTM here:
  - Explicit trajectory: the model directly sees "tile X had 0, 0, 2, 8, 20
    units over the last 5 turns" — no risk of the recurrent gate forgetting it.
  - No BPTT: cleaner PPO training, no truncated backprop.
  - Interpretable: attention weights on historical frames are visualisable.

Historical frame features per tile (4-dim, dynamic fields only):
  [0]  units / MAX_UNITS
  [1]  is_owned_by_player  (acting player's perspective)
  [2]  is_owned_by_enemy
  [3]  is_neutral

Terrain, coordinates, start-tile flags are static — encoded once in the
current frame. Storing them redundantly in every historical frame would
waste capacity without adding signal.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy

from ...engine.types import Board

# 4 dynamic features per tile per historical frame
N_HIST_FEATURES = 4


class HistoryBuffer:
    """
    Circular buffer of the last K board states for one player's perspective.

    Usage::

        buf = HistoryBuffer(k=5)
        buf.reset()                      # call at game start
        buf.push(board)                  # call after each turn resolves
        hist = buf.get_frames()          # list of K Board snapshots,
                                         # oldest first, newest last

    The buffer is pre-filled with copies of the first board state when reset()
    is called so the encoder always receives exactly K frames, even on turn 1.
    """

    def __init__(self, k: int = 5) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self._frames: deque[Board] = deque(maxlen=k)

    def reset(self, initial_board: Board) -> None:
        """
        Initialise the buffer for a new game.

        Pre-fills all K slots with a copy of the initial board so the
        encoder always receives exactly K frames from the very first turn.

        Args:
            initial_board: The board state at game start.
        """
        self._frames.clear()
        snapshot = _shallow_copy_board(initial_board)
        for _ in range(self.k):
            self._frames.append(snapshot)

    def push(self, board: Board) -> None:
        """
        Store the current board state (called after each turn resolves).

        Args:
            board: Current board (a snapshot is taken — caller may mutate freely).
        """
        self._frames.append(_shallow_copy_board(board))

    def get_frames(self) -> list[Board]:
        """
        Return K board snapshots ordered oldest → newest.

        The last element is the most recent push; the first is K turns ago.
        Always returns exactly K frames (pre-filled with initial state if
        the game hasn't run K turns yet).
        """
        return list(self._frames)

    def __len__(self) -> int:
        return len(self._frames)


def _shallow_copy_board(board: Board) -> Board:
    """
    Shallow-copy the board dict with deep-copied Tiles.

    Tiles are mutable dataclasses — we need to freeze their values at this
    point in time without paying the full cost of deepcopy on the whole graph.
    """
    return {k: deepcopy(v) for k, v in board.items()}
