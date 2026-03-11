"""
Reward shaping for HexWar PPO training.

Reward design principles:
  1. Sparse terminal reward: +1 for winning, -1 for losing/eliminated.
  2. Dense tile-progress reward: small bonus each turn for tile count change.
  3. Dense combat-efficiency reward: small bonus for conquests and kills.
  4. Elimination bonus: +0.5 when a player eliminates another.

All dense rewards are scaled to be smaller than the terminal ±1 signal so
the agent prioritises winning over farming kills. The exact scales are
tunable hyperparameters.

Multi-agent note:
  In 6-player self-play, rewards are zero-sum in aggregate (one winner,
  one terminal +1, and the others get -1 or smaller penalties). This
  encourages competitive play without neutral equilibria.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..engine.turn_resolver import MoveEvent
from ..engine.types import Board, Player


@dataclass
class RewardConfig:
    win_reward: float = 1.0
    loss_reward: float = -1.0
    elimination_reward: float = 0.5    # for eliminating another player
    eliminated_penalty: float = -0.5   # for being eliminated mid-game
    tile_progress_scale: float = 0.01  # per tile gained/lost this turn
    conquest_scale: float = 0.02       # per enemy tile conquered
    kill_scale: float = 0.005          # per enemy unit killed


DEFAULT_REWARD_CONFIG = RewardConfig()


def compute_step_reward(
    player_id: str,
    events: list[MoveEvent],
    board_before: Board,
    board_after: Board,
    players_before: list[Player],
    players_after: list[Player],
    winner_id: str | None,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    """
    Compute the reward signal for one player after one turn step.

    Args:
        player_id:      The player we are computing rewards for.
        events:         MoveEvents generated this turn (for any player).
        board_before:   Board state before the turn.
        board_after:    Board state after the turn.
        players_before: Player list before the turn.
        players_after:  Player list after the turn.
        winner_id:      Winner ID if game ended this step, else None.
        config:         Reward hyperparameters.

    Returns:
        Float reward signal for this step.
    """
    reward = 0.0

    # Terminal reward
    if winner_id is not None:
        if winner_id == player_id:
            return config.win_reward
        else:
            return config.loss_reward

    # Check if this player was just eliminated
    was_alive_before = any(
        p.id == player_id and not p.is_eliminated for p in players_before
    )
    is_alive_after = any(
        p.id == player_id and not p.is_eliminated for p in players_after
    )
    if was_alive_before and not is_alive_after:
        return config.eliminated_penalty

    # Tile progress
    tiles_before = sum(1 for t in board_before.values() if t.owner == player_id)
    tiles_after = sum(1 for t in board_after.values() if t.owner == player_id)
    reward += config.tile_progress_scale * (tiles_after - tiles_before)

    # Elimination bonus (we eliminated someone)
    newly_eliminated = [
        p.id for p in players_after
        if p.is_eliminated
        and any(q.id == p.id and not q.is_eliminated for q in players_before)
    ]
    if newly_eliminated:
        # Award if our events contributed (simplification: give credit if we
        # acted this turn and someone was eliminated)
        our_events = [e for e in events if e.player_id == player_id]
        if our_events:
            reward += config.elimination_reward * len(newly_eliminated)

    # Combat events this turn
    for event in events:
        if event.player_id != player_id:
            continue
        if event.kind == "conquer":
            reward += config.conquest_scale
        elif event.kind == "fight":
            # Partial credit for fighting (we did damage even without conquest)
            reward += config.kill_scale * event.units

    return reward
