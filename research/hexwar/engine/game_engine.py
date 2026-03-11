"""
Headless HexWar game engine for AI training and evaluation.

This module provides HexWarEnv: a self-contained simulation that runs a
complete 6-player game to completion. It mirrors the JS gameEngine.ts logic
but exposes a clean Python interface suited for:
  - Rollout collection (AI training)
  - Tournament evaluation
  - Game replay and analysis

Turn structure (mirrors JS):
  1. Each player (in order p0..p5) submits orders and resolves their turn.
  2. After all players move, generate_units() runs for every tile.
  3. Repeat until win condition or max_turns.

The JS game separates "human turn" and "AI turn" phases for the UI. Here,
all players are treated identically — the caller provides an agent callable
for each player.
"""

from __future__ import annotations

from random import Random
from typing import Callable

from .board_generator import generate_board
from .turn_resolver import TurnResult, resolve_player_turn
from .types import (
    PLAYER_COLORS,
    PLAYER_IDS,
    PLAYER_NAMES,
    Board,
    GameResult,
    GameState,
    OrderMap,
    Player,
    PlayerStats,
    PlayerType,
    TurnState,
)
from .unit_generator import generate_units

# ---------------------------------------------------------------------------
# Agent protocol
# ---------------------------------------------------------------------------

# An agent callable receives (board, player_id, players, stats) and returns
# an OrderMap (fromKey → MovementOrder). Return an empty dict to pass.
AgentFn = Callable[[Board, str, list[Player], dict[str, PlayerStats]], OrderMap]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

MAX_TURNS_DEFAULT = 300


class HexWarEnv:
    """
    Headless HexWar game simulation.

    Usage::

        from hexwar.engine.game_engine import HexWarEnv
        from hexwar.agents.random_agent import RandomAgent

        agents = {pid: RandomAgent() for pid in PLAYER_IDS}
        env = HexWarEnv(agents=agents, seed=42)
        result = env.run()
        print(result.winner_id, result.turns_played)

    Attributes:
        board:    Current board state.
        players:  Current player list (including eliminated).
        stats:    Per-player running stats.
        turn:     Current turn state.
        phase:    "playerTurn" | "generateUnits" | "end"
    """

    def __init__(
        self,
        agents: dict[str, AgentFn] | None = None,
        player_ids: list[str] | None = None,
        seed: int | None = None,
        rng: Random | None = None,
        max_turns: int = MAX_TURNS_DEFAULT,
    ) -> None:
        """
        Initialise a new game.

        Args:
            agents:     Dict mapping player_id → AgentFn. Players without
                        an agent pass each turn (no orders).
            player_ids: Override the default 6-player list.
            seed:       RNG seed for reproducible board generation.
            rng:        Pre-constructed Random instance (overrides seed).
            max_turns:  Hard cap; game ends as draw if exceeded.
        """
        self.player_ids = player_ids or PLAYER_IDS
        self.agents: dict[str, AgentFn] = agents or {}
        self.max_turns = max_turns
        self._elimination_order: list[str] = []

        # Generate board
        _rng = rng if rng is not None else Random(seed)
        self.board: Board = generate_board(
            player_ids=self.player_ids,
            rng=_rng,
        )

        # Build players
        self.players: list[Player] = [
            Player(
                id=pid,
                name=PLAYER_NAMES[i] if i < len(PLAYER_NAMES) else pid,
                color=PLAYER_COLORS[i] if i < len(PLAYER_COLORS) else "#ffffff",
                type=PlayerType.AI,
            )
            for i, pid in enumerate(self.player_ids)
        ]

        # Initialise stats
        self.stats: dict[str, PlayerStats] = {
            pid: PlayerStats(player_id=pid)
            for pid in self.player_ids
        }

        self.turn = TurnState(turn_number=1, active_ai_index=0)
        self.phase = "playerTurn"
        self.winner_id: str | None = None
        # Fixed turn order: always iterate over the same player list in order.
        # active_ai_index indexes into self.player_ids (not just live players)
        # so eliminations mid-round don't cause skipping.

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> GameResult:
        """
        Run the game to completion and return the result.

        Returns:
            GameResult with winner, turns played, final stats, and
            elimination order.
        """
        while self.phase != "end":
            self.step()
        return GameResult(
            winner_id=self.winner_id,
            turns_played=self.turn.turn_number - 1,
            stats=self.stats,
            elimination_order=self._elimination_order,
        )

    def step(self) -> None:
        """
        Advance the game by one logical step.

        One step = one player's turn OR the unit-generation phase.
        Callers that want fine-grained control can call step() in a loop
        instead of run().
        """
        if self.phase == "end":
            return

        if self.phase == "playerTurn":
            self._resolve_next_player()
        elif self.phase == "generateUnits":
            self._do_generate_units()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_next_player(self) -> None:
        """Resolve the current player's turn, then advance to the next.

        We iterate over player_ids in fixed order (not just live players) so
        that mid-round eliminations don't cause indices to shift and skip turns.
        """
        idx = self.turn.active_ai_index
        if idx >= len(self.player_ids):
            # All players have had their turn — transition to unit generation
            self.phase = "generateUnits"
            return

        pid = self.player_ids[idx]
        player = next((p for p in self.players if p.id == pid), None)

        # Advance index regardless — the player may be eliminated or missing
        self.turn.active_ai_index += 1

        if player is None or player.is_eliminated:
            return  # Skip eliminated players silently
        agent = self.agents.get(player.id)
        orders: OrderMap = {}
        if agent is not None:
            orders = agent(self.board, player.id, self.players, self.stats)

        result: TurnResult = resolve_player_turn(
            board=self.board,
            players=self.players,
            orders=orders,
            player_id=player.id,
            stats=self.stats,
        )

        self.board = result.board
        self.players = result.players
        self.stats = result.stats

        # Track newly eliminated
        for p in self.players:
            if p.is_eliminated and p.id not in self._elimination_order:
                self._elimination_order.append(p.id)

        if result.winner_id:
            self._end_game(winner_id=result.winner_id)

    def _do_generate_units(self) -> None:
        """Run the unit generation phase and start the next turn."""
        generate_units(self.board, self.stats)

        if self.turn.turn_number >= self.max_turns:
            self._end_game(winner_id=None)
            return

        self.turn = TurnState(
            turn_number=self.turn.turn_number + 1,
            active_ai_index=0,
        )
        self.phase = "playerTurn"

    def _end_game(self, winner_id: str | None) -> None:
        self.winner_id = winner_id
        self.phase = "end"

    # ------------------------------------------------------------------
    # Observation helpers (used by AI agents)
    # ------------------------------------------------------------------

    def get_game_state(self) -> GameState:
        """
        Return a GameState snapshot of the current game.

        This is a convenience method for agents that expect a GameState.
        The returned state is a shallow copy — do not mutate.
        """
        return GameState(
            board=self.board,
            players=self.players,
            human_player_id=self.player_ids[0],
            orders={},
            human_standing_orders={},
            phase=self.phase,
            turn=self.turn,
            stats=self.stats,
        )
