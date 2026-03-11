"""
Core game types — mirrors the TypeScript interfaces in src/types/.

Design notes:
  - All dataclasses use slots=True for memory efficiency (many tiles per game).
  - Board is a plain dict[str, Tile]; key format is "q,r" (matches JS hexToKey).
  - OrderMap maps fromKey → MovementOrder for one player's orders.
  - AllOrders maps playerId → OrderMap for all players in a turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

# ---------------------------------------------------------------------------
# Hex coordinates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AxialCoord:
    q: int
    r: int


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------

TerrainType = Literal["grassland", "plains", "desert", "tundra"]

TERRAIN_TYPES: tuple[TerrainType, ...] = ("grassland", "plains", "desert", "tundra")


# ---------------------------------------------------------------------------
# Tile
# ---------------------------------------------------------------------------

@dataclass()
class Tile:
    coord: AxialCoord
    owner: str | None         # PlayerId or None for neutral
    units: int
    is_start_tile: bool
    start_owner: str | None   # original owner of the start tile
    terrain: TerrainType
    newly_conquered: bool = False


# ---------------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------------

# Board = dict[str, Tile]  — key is "q,r"
Board = dict[str, Tile]


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------

class PlayerType(Enum):
    HUMAN = "human"
    AI = "ai"


@dataclass()
class Player:
    id: str
    name: str
    color: str
    type: PlayerType
    is_eliminated: bool = False


# Canonical player IDs and colors matching the JS game
PLAYER_IDS = ["p1", "p2", "p3", "p4", "p5", "p6"]

PLAYER_COLORS = [
    "#e05c5c",  # red
    "#5c8fe0",  # blue
    "#5ce07a",  # green
    "#e0c05c",  # yellow
    "#c05ce0",  # purple
    "#5ce0d4",  # cyan
]

PLAYER_NAMES = ["Red", "Blue", "Green", "Yellow", "Purple", "Cyan"]


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

UNITS_ALL = -1  # sentinel: send all available units


@dataclass()
class MovementOrder:
    from_key: str
    to_key: str
    requested_units: int  # actual count or UNITS_ALL


# fromKey → MovementOrder for one player
OrderMap = dict[str, MovementOrder]

# playerId → OrderMap for all players
AllOrders = dict[str, OrderMap]


# ---------------------------------------------------------------------------
# Game phase
# ---------------------------------------------------------------------------

GamePhase = Literal["playerTurn", "aiTurn", "generateUnits", "end"]


# ---------------------------------------------------------------------------
# Turn state
# ---------------------------------------------------------------------------

@dataclass()
class TurnState:
    turn_number: int = 1
    active_ai_index: int = 0


# ---------------------------------------------------------------------------
# Per-player statistics
# ---------------------------------------------------------------------------

@dataclass
class PlayerStats:
    player_id: str
    tiles_held: int = 0
    units_generated: int = 0
    units_lost: int = 0
    units_killed: int = 0
    tiles_captured: int = 0
    tiles_lost: int = 0


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    board: Board
    players: list[Player]
    human_player_id: str        # for full-AI sims, treat p1 as "human"
    orders: AllOrders           # current turn's one-off orders
    human_standing_orders: OrderMap  # standing orders (human only in JS; unused in AI sims)
    phase: GamePhase
    turn: TurnState
    stats: dict[str, PlayerStats] = field(default_factory=dict)

    @property
    def live_players(self) -> list[Player]:
        return [p for p in self.players if not p.is_eliminated]

    @property
    def ai_players(self) -> list[Player]:
        return [p for p in self.players if p.type == PlayerType.AI and not p.is_eliminated]


# ---------------------------------------------------------------------------
# Game result (returned after a completed game)
# ---------------------------------------------------------------------------

@dataclass
class GameResult:
    winner_id: str | None          # None if max turns reached with no winner
    turns_played: int
    stats: dict[str, PlayerStats]
    elimination_order: list[str]   # earliest eliminated first
