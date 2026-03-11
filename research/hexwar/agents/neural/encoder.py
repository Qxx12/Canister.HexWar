"""
Board-to-graph encoder for the neural agent, with optional frame stacking.

─────────────────────────────────────────────────────────────────
Single-frame encoding  (encode_board)
─────────────────────────────────────────────────────────────────
Node features (per tile, 18-dim):
  [0]   units / MAX_UNITS
  [1]   is_owned_by_player
  [2]   is_owned_by_enemy
  [3]   is_neutral
  [4]   is_start_tile
  [5]   is_my_start_tile
  [6]   newly_conquered
  [7-10] terrain one-hot (grassland, plains, desert, tundra)
  [11]  n_enemy_neighbors / 6
  [12]  n_friendly_neighbors / 6
  [13]  n_neutral_neighbors / 6
  [14]  q  (normalised)
  [15]  r  (normalised)
  [16]  s = -(q+r)  (normalised)
  [17]  max enemy-neighbor units / MAX_UNITS

─────────────────────────────────────────────────────────────────
Frame-stacked encoding  (encode_board_with_history)
─────────────────────────────────────────────────────────────────
Node features: 18 + (K-1) × 4  dim
  Current frame  : full 18-dim  (as above)
  Each past frame: 4-dim snapshot
    [0]  units / MAX_UNITS
    [1]  is_owned_by_player
    [2]  is_owned_by_enemy
    [3]  is_neutral

Ordering: [current_18 | frame_{t-1}_4 | frame_{t-2}_4 | ... | frame_{t-K+1}_4]

The historical frames give the model explicit unit-count trajectories per
tile, exposing buildup patterns like:
  tile A:  units = [0, 0, 3, 9, 22]   → massing attack incoming
  tile B:  units = [15, 11, 6, 2, 0]  → evacuated toward border
  tile C:  owner = [e, e, e, m, m]    → recently conquered, being reinforced

─────────────────────────────────────────────────────────────────
Edge features (3-dim, unchanged)
  [0]  same_owner
  [1]  enemy_edge
  [2]  border_weight = 1 / (1 + |unit_diff|)

Global features (12-dim, unchanged)
  [0-5]   tile fractions per player (acting player first)
  [6-11]  unit fractions per player
─────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch_geometric.data import Data

from ...engine.board_generator import BOARD_SIZE
from ...engine.hex_utils import hex_neighbors, hex_to_key
from ...engine.types import PLAYER_IDS, Board
from .history_buffer import N_HIST_FEATURES

MAX_UNITS = 50.0
HALF = BOARD_SIZE / 2.0

TERRAIN_ORDER = ("grassland", "plains", "desert", "tundra")

N_NODE_FEATURES   = 18          # current-frame features
N_EDGE_FEATURES   = 3
N_GLOBAL_FEATURES = 12


def node_feature_dim(k: int = 1) -> int:
    """Total node feature dimension for K-frame stacking."""
    return N_NODE_FEATURES + (k - 1) * N_HIST_FEATURES


# ---------------------------------------------------------------------------
# Single-frame encoder (used when no history is available)
# ---------------------------------------------------------------------------

def encode_board(
    board: Board,
    player_id: str,
    all_player_ids: list[str] | None = None,
) -> Data:
    """
    Encode a single board state. Equivalent to encode_board_with_history
    with k=1 (no temporal context).
    """
    return encode_board_with_history(
        frames=[board],
        player_id=player_id,
        all_player_ids=all_player_ids,
    )


# ---------------------------------------------------------------------------
# Multi-frame encoder
# ---------------------------------------------------------------------------

def encode_board_with_history(
    frames: list[Board],
    player_id: str,
    all_player_ids: list[str] | None = None,
) -> Data:
    """
    Encode K board snapshots as a single PyTorch Geometric Data object.

    Args:
        frames:         List of K board snapshots ordered oldest → newest.
                        frames[-1] is the current state. All frames must
                        have the same tile keys.
        player_id:      Acting player's ID.
        all_player_ids: Player ordering for global features.

    Returns:
        torch_geometric.data.Data with:
          x          — node features  [N, 18 + (K-1)×4]
          edge_index — [2, E]
          edge_attr  — [E, 3]
          u          — global features [12]
          node_keys  — list[str] tile keys in node order
          k          — int, number of frames encoded
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as e:
        raise ImportError(
            "torch and torch_geometric are required for the neural agent. "
            "Run: pip install torch torch_geometric"
        ) from e

    if not frames:
        raise ValueError("frames must be non-empty")

    k = len(frames)
    current = frames[-1]   # most recent frame
    history = frames[:-1]  # oldest … second-most-recent

    if all_player_ids is None:
        all_player_ids = PLAYER_IDS

    keys = sorted(current.keys())
    key_to_idx = {key: i for i, key in enumerate(keys)}
    n = len(keys)
    feat_dim = node_feature_dim(k)

    x = torch.zeros(n, feat_dim, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Current-frame features (18-dim)
    # ------------------------------------------------------------------
    for i, key in enumerate(keys):
        tile = current[key]
        owner = tile.owner

        nb_keys = [hex_to_key(nb) for nb in hex_neighbors(tile.coord) if hex_to_key(nb) in current]
        n_enemy    = sum(1 for k_ in nb_keys if current[k_].owner is not None and current[k_].owner != player_id)
        n_friendly = sum(1 for k_ in nb_keys if current[k_].owner == player_id)
        n_neutral  = sum(1 for k_ in nb_keys if current[k_].owner is None)
        enemy_max  = max(
            (current[k_].units for k_ in nb_keys
             if current[k_].owner is not None and current[k_].owner != player_id),
            default=0.0,
        )

        x[i, 0]  = tile.units / MAX_UNITS
        x[i, 1]  = float(owner == player_id)
        x[i, 2]  = float(owner is not None and owner != player_id)
        x[i, 3]  = float(owner is None)
        x[i, 4]  = float(tile.is_start_tile)
        x[i, 5]  = float(tile.is_start_tile and tile.start_owner == player_id)
        x[i, 6]  = float(tile.newly_conquered)
        for j, terrain in enumerate(TERRAIN_ORDER):
            x[i, 7 + j] = float(tile.terrain == terrain)
        x[i, 11] = n_enemy / 6.0
        x[i, 12] = n_friendly / 6.0
        x[i, 13] = n_neutral / 6.0
        x[i, 14] = tile.coord.q / HALF
        x[i, 15] = tile.coord.r / HALF
        x[i, 16] = -(tile.coord.q + tile.coord.r) / HALF
        x[i, 17] = enemy_max / MAX_UNITS

    # ------------------------------------------------------------------
    # Historical frames — oldest first, each 4-dim
    # Stacked after current features, most-recent-history first
    # (i.e. frame t-1 at offset 18, frame t-2 at offset 22, ...)
    # ------------------------------------------------------------------
    for hist_idx, hist_board in enumerate(reversed(history)):
        # hist_idx=0 → frame t-1, hist_idx=1 → frame t-2, ...
        col_start = N_NODE_FEATURES + hist_idx * N_HIST_FEATURES
        for i, key in enumerate(keys):
            tile = hist_board.get(key)
            if tile is None:
                continue   # tile didn't exist yet (shouldn't happen)
            owner = tile.owner
            x[i, col_start + 0] = tile.units / MAX_UNITS
            x[i, col_start + 1] = float(owner == player_id)
            x[i, col_start + 2] = float(owner is not None and owner != player_id)
            x[i, col_start + 3] = float(owner is None)

    # ------------------------------------------------------------------
    # Edges (use current board only — topology is static)
    # ------------------------------------------------------------------
    src_list: list[int] = []
    dst_list: list[int] = []
    edge_feats: list[list[float]] = []

    for i, from_key in enumerate(keys):
        from_tile  = current[from_key]
        from_owner = from_tile.owner

        for nb in hex_neighbors(from_tile.coord):
            to_key = hex_to_key(nb)
            j = key_to_idx.get(to_key)
            if j is None:
                continue
            to_tile  = current[to_key]
            to_owner = to_tile.owner

            same_owner = float(from_owner is not None and from_owner == to_owner)
            enemy_edge = float(
                (from_owner == player_id and to_owner is not None and to_owner != player_id)
                or (to_owner == player_id and from_owner is not None and from_owner != player_id)
            )
            border_weight = 1.0 / (1.0 + abs(from_tile.units - to_tile.units))

            src_list.append(i)
            dst_list.append(j)
            edge_feats.append([same_owner, enemy_edge, border_weight])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(edge_feats, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Global features (current board)
    # ------------------------------------------------------------------
    total_tiles = len(keys)
    total_units = max(sum(t.units for t in current.values()), 1)

    tile_fracs = [sum(1 for t in current.values() if t.owner == pid) / total_tiles
                  for pid in all_player_ids]
    unit_fracs = [sum(t.units for t in current.values() if t.owner == pid) / total_units
                  for pid in all_player_ids]

    pidx = all_player_ids.index(player_id) if player_id in all_player_ids else 0
    tile_fracs = tile_fracs[pidx:] + tile_fracs[:pidx]
    unit_fracs = unit_fracs[pidx:] + unit_fracs[:pidx]

    def _pad6(lst: list[float]) -> list[float]:
        return (lst + [0.0] * 6)[:6]

    u = torch.tensor(_pad6(tile_fracs) + _pad6(unit_fracs), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.u = u
    data.node_keys = keys   # type: ignore[assignment]
    data.k = k              # type: ignore[assignment]
    return data
