"""
Unit generation — faithful Python port of src/engine/unitGenerator.ts.

Each owned tile gains +1 unit per generate-units phase, EXCEPT tiles that
were newly conquered this turn (they are "cooling down" and only have their
newly_conquered flag cleared instead).

This mirrors the JS exactly: newly conquered tiles produce no units in the
same turn they were taken, giving defenders a brief reprieve.
"""

from __future__ import annotations

from .types import Board, PlayerStats


def generate_units(
    board: Board,
    stats: dict[str, PlayerStats],
) -> None:
    """
    Apply end-of-round unit generation to the board in-place.

    For each owned tile:
      - If newly_conquered: clear the flag (no units generated).
      - Otherwise: +1 unit and increment player's unitsGenerated stat.

    Mirrors JS generateUnits().

    Args:
        board: The game board (mutated in-place).
        stats: Per-player stat accumulators (mutated in-place).
    """
    for tile in board.values():
        if tile.owner is None:
            continue
        if tile.newly_conquered:
            tile.newly_conquered = False
        else:
            tile.units += 1
            player_stats = stats.get(tile.owner)
            if player_stats:
                player_stats.units_generated += 1
