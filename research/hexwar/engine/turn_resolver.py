"""
Turn resolution — faithful Python port of src/engine/turnResolver.ts.

Key design decisions preserved from JS:
  - Snapshot cap: only units present at turn START may be ordered.
    This prevents chaining (attacking with units that were just moved in).
  - Orders are processed sequentially in insertion order.
  - Win/elimination is checked after each conquest.
  - Stats accumulation mirrors JS exactly.

The JS returns animation steps (for the frontend). The Python version
returns TurnResult which omits animation-specific fields but retains
all board/player/stats state useful for AI training.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from .combat import apply_combat_result, resolve_combat
from .hex_utils import hex_neighbors, hex_to_key
from .types import UNITS_ALL, Board, OrderMap, Player, PlayerStats
from .win_condition import check_eliminations, check_win

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class MoveEvent:
    """Lightweight record of a single resolved move (analogous to AnimationEvent)."""
    kind: str            # "move" | "fight" | "conquer"
    from_key: str
    to_key: str
    player_id: str
    units: int


@dataclass
class TurnResult:
    board: Board
    players: list[Player]
    events: list[MoveEvent]
    winner_id: str | None
    stats: dict[str, PlayerStats]


# ---------------------------------------------------------------------------
# Core resolver
# ---------------------------------------------------------------------------

def resolve_player_turn(
    board: Board,
    players: list[Player],
    orders: OrderMap,
    player_id: str,
    stats: dict[str, PlayerStats],
) -> TurnResult:
    """
    Execute one player's orders against the board.

    Mirrors JS resolvePlayerTurn().

    Args:
        board:     Current board state (will be copied and mutated).
        players:   Current player list (will be copied).
        orders:    This player's OrderMap (fromKey → MovementOrder).
        player_id: The acting player's ID.
        stats:     Running stats dict (will be copied and updated).

    Returns:
        TurnResult with the updated board, players, events, winner, and stats.
    """
    current_board: Board = {k: deepcopy(v) for k, v in board.items()}
    initial_board: Board = {k: deepcopy(v) for k, v in board.items()}
    current_players = [deepcopy(p) for p in players]
    current_stats = {k: deepcopy(v) for k, v in stats.items()}
    events: list[MoveEvent] = []

    for from_key, order in orders.items():
        from_tile = current_board.get(from_key)
        if from_tile is None or from_tile.owner != player_id or from_tile.units == 0:
            continue

        to_tile = current_board.get(order.to_key)
        if to_tile is None:
            continue

        # Validate adjacency
        neighbor_keys = [hex_to_key(n) for n in hex_neighbors(from_tile.coord)]
        if order.to_key not in neighbor_keys:
            continue

        # Snapshot cap: limit to units available at turn start.
        # Resolve UNITS_ALL (-1) sentinel to "all initial units".
        initial_units = initial_board[from_key].units if from_key in initial_board else 0
        effective_requested = (
            initial_units if order.requested_units == UNITS_ALL
            else order.requested_units
        )
        capped_requested = min(effective_requested, initial_units)
        if capped_requested <= 0:
            continue

        result = resolve_combat(
            current_board,
            from_key,
            order.to_key,
            capped_requested,
            player_id,
        )
        if result.units_sent == 0:
            continue

        # Update stats
        atk_stats = current_stats.get(player_id)
        if result.defender_casualties > 0 and result.defender_id and atk_stats:
            atk_stats.units_killed += result.defender_casualties

        if result.conquered:
            if atk_stats:
                atk_stats.tiles_captured += 1
            if result.defender_id:
                def_stats = current_stats.get(result.defender_id)
                if def_stats:
                    def_stats.tiles_lost += 1

        # Mutate board
        apply_combat_result(current_board, result)

        # Classify event
        if result.defender_casualties > 0:
            kind = "conquer" if result.conquered else "fight"
        else:
            kind = "move"

        events.append(MoveEvent(
            kind=kind,
            from_key=from_key,
            to_key=order.to_key,
            player_id=player_id,
            units=result.units_sent,
        ))

        # Check win/elimination after each conquest
        winner_id: str | None = None
        if result.conquered:
            winner_id = check_win(current_board, current_players)
            if not winner_id:
                newly_eliminated = check_eliminations(current_board, current_players)
                for p in current_players:
                    if p.id in newly_eliminated:
                        p.is_eliminated = True

        if winner_id:
            return TurnResult(
                board=current_board,
                players=current_players,
                events=events,
                winner_id=winner_id,
                stats=current_stats,
            )

    return TurnResult(
        board=current_board,
        players=current_players,
        events=events,
        winner_id=None,
        stats=current_stats,
    )
