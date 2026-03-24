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

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from ..engine.hex_utils import hex_neighbors, hex_to_key
from ..engine.types import Board, MovementOrder, OrderMap, Player, PlayerStats
from .base_agent import BaseAgent

# Players with this many tiles or fewer are near-elimination.
# Finishing them removes an entire front permanently.
NEAR_ELIM_THRESHOLD = 4

# Default weight vector (11-dim) — good starting point for CMA-ES
#
# Key constraint: attack_enemy alone (when can_conquer=0) must score BELOW
# expand_neutral to prevent suicidal attacks.
#   non-winning attack score = 0*can_conquer + attack_enemy = 0.1
#   neutral expansion score  = expand_neutral = 1.0
#   winning attack score     = 1*can_conquer + attack_enemy = 2.1   > neutral ✓
#
# Features 8-10 are new in v4:
#   8  inv_dist_to_unowned_start — BFS gradient toward win-condition tiles
#   9  target_owner_near_elim    — bonus for finishing off a weak player
#  10  neutral_adj_to_target     — expansion value: how many new neutrals open up
DEFAULT_WEIGHTS: tuple[float, ...] = (
    2.0,   # 0  can_conquer
    4.0,   # 1  is_start_tile
    1.0,   # 2  expand_neutral
    0.1,   # 3  attack_enemy
    1.0,   # 4  units_advantage
    0.3,   # 5  relative_tile_count
    0.1,   # 6  border_exposure
    0.5,   # 7  reinforce_friendly
    0.0,   # 8  inv_dist_to_unowned_start  (disabled; CMA-ES discovers value via warmstart)
    0.0,   # 9  target_owner_near_elim     (disabled; CMA-ES discovers value via warmstart)
    0.0,   # 10 neutral_adj_to_target      (disabled; CMA-ES discovers value via warmstart)
)

N_FEATURES = len(DEFAULT_WEIGHTS)   # 11


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

        # Precompute once per turn and passed into every _score_move call.
        start_gradient: dict[str, float] = self._compute_start_gradient(board, player_id)
        tile_counts: dict[str, int] = {}
        for _t in board.values():
            if _t.owner:
                tile_counts[_t.owner] = tile_counts.get(_t.owner, 0) + 1

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
                    start_gradient=start_gradient,
                    tile_counts=tile_counts,
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
        start_gradient: dict[str, float],
        tile_counts: dict[str, int],
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
        n_theirs = tile_counts.get(defender_id, 0) if is_hostile else 0
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

        # Feature 8: inv_dist_to_unowned_start
        # BFS inverse-distance from this target tile to the nearest unowned
        # start tile. 1.0 if target IS an unowned start tile (dist=0), 0.5 if
        # one hop away, 0.33 at two hops, etc. Encodes the win-condition path.
        inv_dist_to_start = start_gradient.get(to_key, 0.0)

        # Feature 9: target_owner_near_elim
        # Bonus for attacking a player with ≤ NEAR_ELIM_THRESHOLD tiles.
        # Eliminating a player collapses their entire front permanently.
        target_owner_near_elim = float(
            is_hostile
            and tile_counts.get(defender_id, 0) <= NEAR_ELIM_THRESHOLD
        )

        # Feature 10: neutral_adj_to_target
        # Count of neutral tiles adjacent to the target tile, normalised to
        # [0, 1]. Capturing a "junction" tile that opens multiple neutral
        # paths is worth more than capturing a dead-end tile.
        to_neighbors = hex_neighbors(to_tile.coord)
        n_neutral_adj = sum(
            1 for nb in to_neighbors
            if (nt := board.get(hex_to_key(nb))) is not None and nt.owner is None
        )
        neutral_adj_to_target = n_neutral_adj / 6.0

        features = [
            can_conquer,
            is_start,
            expand_neutral,
            attack_enemy,
            units_advantage,
            relative_tc,
            border_exposure,
            reinforce,
            inv_dist_to_start,
            target_owner_near_elim,
            neutral_adj_to_target,
        ]

        return sum(w * f for w, f in zip(self._weights, features))

    # ------------------------------------------------------------------
    # BFS helper
    # ------------------------------------------------------------------

    def _compute_start_gradient(
        self, board: Board, player_id: str
    ) -> dict[str, float]:
        """
        Multi-source BFS from every unowned start tile.

        Returns {tile_key: 1/(1+dist)} for all board-reachable tiles.
          dist=0 at an unowned start tile  → inv_dist = 1.0
          dist=1 at its neighbours         → inv_dist = 0.5
          dist=2                           → inv_dist = 0.33  etc.

        Returns an empty dict when all start tiles are owned by us
        (win is imminent; gradient has no meaning).
        """
        seeds = [
            key for key, t in board.items()
            if t.is_start_tile and t.owner != player_id
        ]
        if not seeds:
            return {}

        dist: dict[str, int] = {}
        q: deque[str] = deque()
        for key in seeds:
            dist[key] = 0
            q.append(key)

        while q:
            key = q.popleft()
            tile = board.get(key)
            if tile is None:
                continue
            for nb in hex_neighbors(tile.coord):
                nb_key = hex_to_key(nb)
                if nb_key in board and nb_key not in dist:
                    dist[nb_key] = dist[key] + 1
                    q.append(nb_key)

        return {key: 1.0 / (1.0 + d) for key, d in dist.items()}

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
