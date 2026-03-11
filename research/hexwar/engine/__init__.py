"""Faithful Python port of the HexWar JS game engine."""

from .board_generator import generate_board
from .combat import CombatResult, apply_combat_result, resolve_combat
from .game_engine import AgentFn, HexWarEnv
from .hex_utils import (
    axial_to_pixel,
    hex_distance,
    hex_neighbor_keys,
    hex_neighbors,
    hex_to_key,
    key_to_hex,
)
from .turn_resolver import MoveEvent, TurnResult, resolve_player_turn
from .types import (
    PLAYER_COLORS,
    PLAYER_IDS,
    PLAYER_NAMES,
    TERRAIN_TYPES,
    UNITS_ALL,
    AllOrders,
    AxialCoord,
    Board,
    GamePhase,
    GameResult,
    GameState,
    MovementOrder,
    OrderMap,
    Player,
    PlayerStats,
    PlayerType,
    TerrainType,
    Tile,
    TurnState,
)
from .unit_generator import generate_units
from .win_condition import check_eliminations, check_win

__all__ = [
    # types
    "AxialCoord", "Tile", "Board", "Player", "PlayerType",
    "MovementOrder", "OrderMap", "AllOrders",
    "GamePhase", "TurnState", "PlayerStats", "GameState", "GameResult",
    "PLAYER_IDS", "PLAYER_COLORS", "PLAYER_NAMES",
    "TerrainType", "TERRAIN_TYPES", "UNITS_ALL",
    # hex utils
    "hex_to_key", "key_to_hex", "hex_neighbors", "hex_neighbor_keys",
    "hex_distance", "axial_to_pixel",
    # engine
    "generate_board",
    "CombatResult", "resolve_combat", "apply_combat_result",
    "generate_units",
    "check_win", "check_eliminations",
    "TurnResult", "MoveEvent", "resolve_player_turn",
    "HexWarEnv", "AgentFn",
]
