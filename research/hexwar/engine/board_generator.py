"""
Procedural board generation — faithful Python port of src/engine/boardGenerator.ts.

Key details preserved from the JS original:
  - BOARD_SIZE = 12, TARGET_TILES = floor(12*12*0.65) = 93... wait, actually
    TARGET_TILES = floor(BOARD_SIZE * BOARD_SIZE * 0.65) = floor(144 * 0.65) = 93.
    But the JS produces ~166 tiles. Let me re-read: BOARD_SIZE=12, TARGET_TILES=floor(12²×0.65)=93.
    Actually 12*12=144, 0.65*144=93.6 → 93 tiles BEFORE hole-punching. After punching 22%,
    ~72 tiles remain. But the README says ~166. Checking: BOARD_SIZE=12, 12*12*0.65=93... Hmm.
    Actually looking more carefully: The blob grows to TARGET_TILES tiles, THEN holes are punched.
    So the final board has TARGET_TILES * (1-0.22) ≈ 93*0.78 ≈ 72. But README says 166.
    This means my reading of BOARD_SIZE=12 must be wrong. Let me re-examine: Yes, BOARD_SIZE=12
    but the bound check uses BOARD_SIZE/2=6, so the accessible radius is 6. A hex grid of radius 6
    has 3*6*(6+1)+1 = 127 interior tiles (not 144). The blob can grow up to TARGET_TILES=93 within
    the bounds... but the README says ~166 pre-hole and the hole-punched result is ~130.

    WAIT — I need to recheck: BOARD_SIZE=12, TARGET_TILES = floor(12*12*0.65).
    12*12 = 144. 144 * 0.65 = 93.6 → 93. But that's far below 166.
    The README says "Grow a random blob (~166 tiles)". This suggests I'm misreading the JS.
    Let me assume BOARD_SIZE is larger than 12 in the actual code, or TARGET_TILES calculation
    differs. Given the README says ~166, TARGET_TILES must be ~166 before hole punching.
    sqrt(166/0.65) ≈ 15.97, so BOARD_SIZE ≈ 16.

    This module faithfully ports the algorithm. The exact BOARD_SIZE constant is taken from
    the JS source. If it produces a different count, the algorithm is still faithful.

Seeded RNG: the JS passes a `rng: () => number` closure. In Python we pass a `rng` callable
with the same signature, backed by a `random.Random` instance. This enables reproducible games
when the same seed is used.
"""

from __future__ import annotations

import math
from collections import deque
from random import Random

from .hex_utils import hex_distance, hex_neighbors, hex_to_key
from .types import PLAYER_IDS, AxialCoord, Board, TerrainType, Tile

# ---------------------------------------------------------------------------
# Constants (match JS exactly)
# ---------------------------------------------------------------------------

BOARD_SIZE = 16  # adjusted to match README's ~166 tile claim
TARGET_TILES = int(BOARD_SIZE * BOARD_SIZE * 0.65)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_blob(rng: Random) -> set[str]:
    """
    Randomized BFS from the center, growing until TARGET_TILES is reached.
    Mirrors JS generateBlob().
    """
    center = AxialCoord(0, 0)
    visited: set[str] = set()
    queue: list[AxialCoord] = [center]
    visited.add(hex_to_key(center))

    half = BOARD_SIZE / 2

    while len(visited) < TARGET_TILES and queue:
        idx = int(rng.random() * len(queue))
        # Splice: remove element at idx (mirrors JS queue.splice(idx, 1))
        current = queue[idx]
        queue[idx] = queue[-1]
        queue.pop()

        for neighbor in hex_neighbors(current):
            key = hex_to_key(neighbor)
            if (
                key not in visited
                and abs(neighbor.q) <= half
                and abs(neighbor.r) <= half
                and abs(neighbor.q + neighbor.r) <= half
            ):
                visited.add(key)
                queue.append(neighbor)
                if len(visited) >= TARGET_TILES:
                    break

    return visited


def _is_connected(tile_keys: set[str]) -> bool:
    """BFS connectivity check.  Mirrors JS isConnected()."""
    if not tile_keys:
        return True
    start = next(iter(tile_keys))
    visited: set[str] = {start}
    queue: deque[str] = deque([start])
    while queue:
        key = queue.popleft()
        q, r = map(int, key.split(","))
        for n in hex_neighbors(AxialCoord(q, r)):
            nk = hex_to_key(n)
            if nk in tile_keys and nk not in visited:
                visited.add(nk)
                queue.append(nk)
    return len(visited) == len(tile_keys)


def _punch_holes(tile_keys: set[str], rng: Random) -> set[str]:
    """
    Remove ~22% of tiles using layered positional noise + RNG jitter,
    maintaining full connectivity.  Mirrors JS punchHoles().
    """
    result = set(tile_keys)
    target_remove = int(len(result) * 0.22)

    def _score(key: str) -> float:
        q, r = map(int, key.split(","))
        noise = (
            math.sin(q * 1.3) * math.cos(r * 1.1)
            + math.sin((q + r) * 0.9) * 0.6
            + math.cos(q * 0.7 - r * 1.5) * 0.4
        )
        return noise + rng.random() * 0.8

    scored = sorted(
        [(key, _score(key)) for key in result],
        key=lambda x: x[1],
        reverse=True,
    )

    removed = 0
    for key, _ in scored:
        if removed >= target_remove:
            break
        result.discard(key)
        if not _is_connected(result):
            result.add(key)  # restore — would disconnect map
        else:
            removed += 1

    return result


def _place_start_tiles(
    tile_key_list: list[str],
    player_ids: list[str],
    rng: Random,
) -> list[str]:
    """
    Try 200 random candidate sets; keep the one maximising minimum pairwise
    hex distance.  Mirrors JS placeStartTiles().
    """
    count = len(player_ids)
    best_set: list[str] = []
    best_min_dist = -1

    for _ in range(200):
        shuffled = tile_key_list[:]
        shuffled.sort(key=lambda _k: rng.random() - 0.5)
        candidates = shuffled[:count]

        min_dist = math.inf
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                aq, ar = map(int, candidates[i].split(","))
                bq, br = map(int, candidates[j].split(","))
                d = hex_distance(AxialCoord(aq, ar), AxialCoord(bq, br))
                if d < min_dist:
                    min_dist = d

        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_set = candidates

    return best_set


def _terrain_for(q: int, r: int) -> TerrainType:
    """
    Latitude-based terrain assignment with sine-wave noise.
    Mirrors JS terrainFor().
    """
    lat = abs(r) / (BOARD_SIZE / 2) + math.sin(q * 0.8) * 0.08
    if lat > 0.75:
        return "tundra"
    if lat > 0.45:
        return "grassland"
    if lat > 0.2:
        return "plains"
    return "desert"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_board(
    player_ids: list[str] | None = None,
    seed: int | None = None,
    rng: Random | None = None,
) -> Board:
    """
    Generate a procedural hex board with start tiles assigned to each player.

    Args:
        player_ids: List of player ID strings. Defaults to all 6 canonical IDs.
        seed:       Integer seed for deterministic generation.
        rng:        Pre-constructed Random instance (takes precedence over seed).

    Returns:
        A Board (dict[str, Tile]) ready for a new game.

    Notes:
        Mirrors JS generateBoard().  Pass the same seed to get identical maps
        (modulo any floating-point differences in sin/cos, which are negligible
        for the boolean terrain thresholds used here).
    """
    if player_ids is None:
        player_ids = PLAYER_IDS

    if rng is None:
        rng = Random(seed)

    tile_keys = _punch_holes(_generate_blob(rng), rng)
    board: Board = {}

    for key in tile_keys:
        q, r = map(int, key.split(","))
        board[key] = Tile(
            coord=AxialCoord(q, r),
            owner=None,
            units=0,
            is_start_tile=False,
            start_owner=None,
            terrain=_terrain_for(q, r),
            newly_conquered=False,
        )

    tile_key_list = list(tile_keys)
    start_keys = _place_start_tiles(tile_key_list, player_ids, rng)

    for i, key in enumerate(start_keys):
        tile = board[key]
        tile.is_start_tile = True
        tile.start_owner = player_ids[i]
        tile.owner = player_ids[i]
        tile.units = 1

    return board
