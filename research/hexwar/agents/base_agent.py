"""
Base agent interface.

All HexWar agents implement __call__(board, player_id, players, stats) →
OrderMap. This matches the AgentFn protocol used by HexWarEnv, so any
agent instance can be passed directly as an agent callable.

Subclasses may override reset() to clear per-game state between episodes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..engine.types import Board, OrderMap, Player, PlayerStats


class BaseAgent(ABC):
    """Abstract base class for all HexWar agents."""

    @abstractmethod
    def __call__(
        self,
        board: Board,
        player_id: str,
        players: list[Player],
        stats: dict[str, PlayerStats],
    ) -> OrderMap:
        """
        Produce orders for one player's turn.

        Args:
            board:     Current board state (read-only — do not mutate).
            player_id: The ID of the player we are acting for.
            players:   Full player list (including eliminated players).
            stats:     Running per-player statistics.

        Returns:
            OrderMap mapping fromKey → MovementOrder.
            Return an empty dict to pass (no moves).
        """
        ...

    def reset(self, initial_board: Board | None = None) -> None:
        """
        Called between episodes to reset any internal state.

        Override in stateful agents (e.g. those that maintain memory across
        turns within a game). Default implementation is a no-op.
        """
        _ = initial_board  # subclasses may use this
