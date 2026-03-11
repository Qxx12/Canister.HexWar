"""
Win and elimination checking — faithful Python port of src/engine/winCondition.ts.

Win condition: a player wins if they own ALL start tiles on the board
AND still own their own starting tile.

Elimination: a player is eliminated if they own zero tiles (and are not
already marked eliminated).
"""

from __future__ import annotations

from .types import Board, Player


def check_win(board: Board, players: list[Player]) -> str | None:
    """
    Return the winner's player_id, or None if no winner yet.

    A player wins by owning every start tile on the board (including their
    own). They must also still hold their own start tile.

    Mirrors JS checkWin(board, players).

    Args:
        board:   Current board state.
        players: Current player list (includes eliminated players).

    Returns:
        Winner's player_id string, or None.
    """
    active = [p for p in players if not p.is_eliminated]
    all_start_tiles = [t for t in board.values() if t.is_start_tile]

    for player in active:
        # Must own their own start tile
        own_start = next(
            (t for t in all_start_tiles if t.start_owner == player.id),
            None,
        )
        if own_start is None or own_start.owner != player.id:
            continue

        # Must own all start tiles
        if all(t.owner == player.id for t in all_start_tiles):
            return player.id

    return None


def check_eliminations(board: Board, players: list[Player]) -> list[str]:
    """
    Return IDs of players who should be eliminated this step.

    A player is eliminated if they own no tiles on the board (and were not
    already eliminated).

    Mirrors JS checkEliminations(board, players).

    Args:
        board:   Current board state.
        players: Current player list.

    Returns:
        List of player IDs to eliminate (may be empty).
    """
    owned: set[str] = {t.owner for t in board.values() if t.owner is not None}
    return [
        p.id
        for p in players
        if not p.is_eliminated and p.id not in owned
    ]
