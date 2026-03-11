"""
Greedy heuristic agent — tunable via a weight vector.

Decision model:
  For each owned tile with units, score every valid adjacent target and
  issue the highest-scoring order. Scores are a weighted sum of hand-crafted
  features. The weight vector is exposed so it can be optimised by CMA-ES.

Features (per candidate move from_tile → to_tile):
  0  can_conquer          1 if units_sent > to_tile.units (hostile), else 0
  1  is_start_tile        1 if to_tile.is_start_tile
  2  expand_neutral        1 if to_tile.owner is None
  3  attack_enemy          1 if to_tile.owner is not None and not ours
  4  units_advantage       (units_sent - to_tile.units) / max(units_sent, 1)
  5  relative_tile_count   (our tiles - their tiles) / total_tiles
  6  border_exposure       # enemy neighbors of from_tile / 6
  7  reinforce_friendly    1 if to_tile.owner == player_id

The default weights encode the following priority order:
  conquer start > conquer normal > expand neutral > reinforce > random
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..engine.hex_utils import hex_neighbors, hex_to_key
from ..engine.types import Board, MovementOrder, OrderMap, Player, PlayerStats
from .base_agent import BaseAgent

# Default weight vector (8-dim) — good starting point for CMA-ES
#
# Key constraint: attack_enemy alone (when can_conquer=0) must score BELOW
# expand_neutral to prevent suicidal attacks.
#   non-winning attack score = 0*can_conquer + attack_enemy = 0.1
#   neutral expansion score  = expand_neutral = 1.0
#   winning attack score     = 1*can_conquer + attack_enemy = 2.1   > neutral ✓
DEFAULT_WEIGHTS: tuple[float, ...] = (
    2.0,   # can_conquer       (dominant when attack wins)
    4.0,   # is_start_tile     (huge bonus for capturing start tiles)
    1.0,   # expand_neutral    (baseline: always expand into empty space)
    0.1,   # attack_enemy      (tiny — suicidal attacks must lose to neutral)
    1.0,   # units_advantage   (prefer attacks with larger margin)
    0.3,   # relative_tile_count
    0.1,   # border_exposure
    0.5,   # reinforce_friendly
)

N_FEATURES = len(DEFAULT_WEIGHTS)


@dataclass
class _Move:
    from_key: str
    to_key: str
    score: float


class GreedyAgent(BaseAgent):
    """
    Greedy agent that scores moves with a linear feature model.

    Args:
        weights: Weight vector of length N_FEATURES (8). Falls back to
                 DEFAULT_WEIGHTS if not provided.
        send_fraction: Fraction of available units to send (default 1.0 =
                       all units). Values in (0, 1] for defensive play.
    """

    def __init__(
        self,
        weights: Sequence[float] | None = None,
        send_fraction: float = 1.0,
    ) -> None:
        if weights is not None and len(weights) != N_FEATURES:
            raise ValueError(f"weights must have length {N_FEATURES}, got {len(weights)}")
        self._weights = list(weights) if weights is not None else list(DEFAULT_WEIGHTS)
        self._send_fraction = max(0.01, min(1.0, send_fraction))

    # ------------------------------------------------------------------
    # AgentFn protocol
    # ------------------------------------------------------------------

    def __call__(
        self,
        board: Board,
        player_id: str,
        players: list[Player],
        stats: dict[str, PlayerStats],
    ) -> OrderMap:
        total_tiles = len(board)
        our_tiles = [k for k, t in board.items() if t.owner == player_id]
        n_ours = len(our_tiles)

        orders: OrderMap = {}

        for from_key in our_tiles:
            from_tile = board[from_key]
            if from_tile.units == 0:
                continue

            units_available = from_tile.units

            # How many units to actually send (never leave start tile empty)
            units_to_send = max(1, round(units_available * self._send_fraction))
            if from_tile.is_start_tile and from_tile.start_owner == player_id:
                # Retain at least 1 unit on our own start tile for defense
                units_to_send = min(units_to_send, max(0, units_available - 1))
            if units_to_send == 0:
                continue

            best: _Move | None = None

            for neighbor in hex_neighbors(from_tile.coord):
                to_key = hex_to_key(neighbor)
                to_tile = board.get(to_key)
                if to_tile is None:
                    continue

                # Skip reinforcing a friendly tile unless that tile has non-owned
                # neighbors (i.e., it's a frontier tile). This prevents oscillation
                # where interior owned tiles bounce units back and forth.
                if to_tile.owner == player_id:
                    nb_keys_of_dest = [hex_to_key(n) for n in hex_neighbors(to_tile.coord)]
                    dest_has_frontier = any(
                        (t := board.get(k)) is not None and t.owner != player_id
                        for k in nb_keys_of_dest
                    )
                    if not dest_has_frontier:
                        continue

                score = self._score_move(
                    from_tile=from_tile,
                    to_tile=to_tile,
                    from_key=from_key,
                    to_key=to_key,
                    player_id=player_id,
                    units_to_send=units_to_send,
                    board=board,
                    n_ours=n_ours,
                    total_tiles=total_tiles,
                )

                if best is None or score > best.score:
                    best = _Move(from_key=from_key, to_key=to_key, score=score)

            if best is not None:
                orders[from_key] = MovementOrder(
                    from_key=from_key,
                    to_key=best.to_key,
                    requested_units=units_to_send,
                )

        return orders

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_move(
        self,
        from_tile,
        to_tile,
        from_key: str,
        to_key: str,
        player_id: str,
        units_to_send: int,
        board: Board,
        n_ours: int,
        total_tiles: int,
    ) -> float:
        defender_id = to_tile.owner
        defending_units = to_tile.units

        # Feature 0: can_conquer (hostile tile, we have more units)
        is_hostile = (defender_id is not None and defender_id != player_id)
        can_conquer = float(is_hostile and units_to_send > defending_units)

        # Feature 1: is_start_tile
        is_start = float(to_tile.is_start_tile)

        # Feature 2: expand_neutral
        expand_neutral = float(defender_id is None)

        # Feature 3: attack_enemy
        attack_enemy = float(is_hostile)

        # Feature 4: units_advantage
        if is_hostile:
            units_advantage = (units_to_send - defending_units) / max(units_to_send, 1)
        else:
            units_advantage = 0.0

        # Feature 5: relative_tile_count (board control)
        if defender_id is not None and defender_id != player_id:
            n_theirs = sum(1 for t in board.values() if t.owner == defender_id)
        else:
            n_theirs = 0
        relative_tc = (n_ours - n_theirs) / max(total_tiles, 1)

        # Feature 6: border_exposure of source tile
        from_neighbors = hex_neighbors(from_tile.coord)
        n_enemy_neighbors = sum(
            1 for n in from_neighbors
            if (t := board.get(hex_to_key(n))) is not None
            and t.owner is not None
            and t.owner != player_id
        )
        border_exposure = n_enemy_neighbors / 6.0

        # Feature 7: reinforce_friendly
        reinforce = float(defender_id == player_id)

        features = [
            can_conquer,
            is_start,
            expand_neutral,
            attack_enemy,
            units_advantage,
            relative_tc,
            border_exposure,
            reinforce,
        ]

        return sum(w * f for w, f in zip(self._weights, features))

    # ------------------------------------------------------------------
    # Weight access (for CMA-ES optimiser)
    # ------------------------------------------------------------------

    @property
    def weights(self) -> list[float]:
        return list(self._weights)

    @weights.setter
    def weights(self, value: Sequence[float]) -> None:
        if len(value) != N_FEATURES:
            raise ValueError(f"Expected {N_FEATURES} weights, got {len(value)}")
        self._weights = list(value)
