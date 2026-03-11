"""
Random agent — baseline for evaluation and sanity testing.

Each turn: for every owned tile that has units, pick a random adjacent tile
and issue a "send all" order. Orders that target non-adjacent tiles are
filtered by the turn resolver, so we rely on that for correctness.

This agent is intentionally simple. It establishes a performance floor
that stronger agents must beat.
"""

from __future__ import annotations

from random import Random

from ..engine.hex_utils import hex_neighbors, hex_to_key
from ..engine.types import UNITS_ALL, Board, MovementOrder, OrderMap, Player, PlayerStats
from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Issues a random valid order from every owned tile that has units."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = Random(seed)

    def __call__(
        self,
        board: Board,
        player_id: str,
        players: list[Player],
        stats: dict[str, PlayerStats],
    ) -> OrderMap:
        orders: OrderMap = {}
        for key, tile in board.items():
            if tile.owner != player_id or tile.units == 0:
                continue
            # Collect neighboring tiles that exist on the board
            neighbors = [
                hex_to_key(n)
                for n in hex_neighbors(tile.coord)
                if hex_to_key(n) in board
            ]
            if not neighbors:
                continue
            target = self._rng.choice(neighbors)
            orders[key] = MovementOrder(
                from_key=key,
                to_key=target,
                requested_units=UNITS_ALL,
            )
        return orders
