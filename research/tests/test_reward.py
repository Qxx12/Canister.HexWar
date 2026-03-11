"""Tests for the PPO reward shaping function."""

from __future__ import annotations

import pytest

from hexwar.engine.turn_resolver import MoveEvent
from hexwar.engine.types import AxialCoord, Player, PlayerType, Tile
from hexwar.training.reward import DEFAULT_REWARD_CONFIG, RewardConfig, compute_step_reward

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_player(pid: str, eliminated: bool = False) -> Player:
    return Player(id=pid, name=pid, color="#000", type=PlayerType.AI,
                  is_eliminated=eliminated)


def _make_tile(owner: str | None, units: int = 5) -> Tile:
    return Tile(
        coord=AxialCoord(0, 0),
        owner=owner,
        units=units,
        is_start_tile=False,
        start_owner=None,
        terrain="grassland",
    )


def _board(mapping: dict[str, str | None]) -> dict:
    """Build a minimal board from {key: owner} mapping."""
    return {k: _make_tile(v) for k, v in mapping.items()}


# ---------------------------------------------------------------------------
# Terminal rewards
# ---------------------------------------------------------------------------

class TestTerminalRewards:
    def test_winner_gets_win_reward(self):
        board = _board({"0,0": "p1"})
        players = [_make_player("p1"), _make_player("p2")]
        r = compute_step_reward("p1", [], board, board, players, players, winner_id="p1")
        assert r == pytest.approx(DEFAULT_REWARD_CONFIG.win_reward)

    def test_loser_gets_loss_reward(self):
        board = _board({"0,0": "p1"})
        players = [_make_player("p1"), _make_player("p2")]
        r = compute_step_reward("p2", [], board, board, players, players, winner_id="p1")
        assert r == pytest.approx(DEFAULT_REWARD_CONFIG.loss_reward)

    def test_terminal_ignores_dense(self):
        """When there is a winner, dense signals are irrelevant."""
        board_before = _board({"0,0": "p2", "1,0": None})
        board_after  = _board({"0,0": "p1", "1,0": "p1"})
        players = [_make_player("p1"), _make_player("p2")]
        r = compute_step_reward("p1", [], board_before, board_after, players, players, winner_id="p1")
        assert r == pytest.approx(DEFAULT_REWARD_CONFIG.win_reward)


# ---------------------------------------------------------------------------
# Elimination penalty
# ---------------------------------------------------------------------------

class TestEliminationPenalty:
    def test_eliminated_player_gets_penalty(self):
        board = _board({"0,0": "p2"})
        before = [_make_player("p1"), _make_player("p2")]
        after  = [_make_player("p1"), _make_player("p2", eliminated=True)]
        r = compute_step_reward("p2", [], board, board, before, after, winner_id=None)
        assert r == pytest.approx(DEFAULT_REWARD_CONFIG.eliminated_penalty)

    def test_survivor_not_penalised(self):
        board = _board({"0,0": "p1"})
        before = [_make_player("p1"), _make_player("p2")]
        after  = [_make_player("p1"), _make_player("p2", eliminated=True)]
        r = compute_step_reward("p1", [], board, board, before, after, winner_id=None)
        assert r != pytest.approx(DEFAULT_REWARD_CONFIG.eliminated_penalty)


# ---------------------------------------------------------------------------
# Tile progress
# ---------------------------------------------------------------------------

class TestTileProgress:
    def test_tile_gain_gives_positive_reward(self):
        before = _board({"0,0": "p1"})
        after  = _board({"0,0": "p1", "1,0": "p1"})
        players = [_make_player("p1"), _make_player("p2")]
        r = compute_step_reward("p1", [], before, after, players, players, winner_id=None)
        assert r > 0.0

    def test_tile_loss_gives_negative_reward(self):
        before = _board({"0,0": "p1", "1,0": "p1"})
        after  = _board({"0,0": "p1", "1,0": "p2"})
        players = [_make_player("p1"), _make_player("p2")]
        r = compute_step_reward("p1", [], before, after, players, players, winner_id=None)
        assert r < 0.0

    def test_no_tile_change_zero_progress(self):
        board = _board({"0,0": "p1"})
        players = [_make_player("p1"), _make_player("p2")]
        r = compute_step_reward("p1", [], board, board, players, players, winner_id=None)
        assert r == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Combat rewards
# ---------------------------------------------------------------------------

class TestCombatRewards:
    def test_conquer_event_gives_bonus(self):
        board = _board({"0,0": "p1", "1,0": "p2"})
        players = [_make_player("p1"), _make_player("p2")]
        event = MoveEvent(kind="conquer", from_key="0,0", to_key="1,0", player_id="p1", units=3)
        r = compute_step_reward("p1", [event], board, board, players, players, winner_id=None)
        assert r == pytest.approx(DEFAULT_REWARD_CONFIG.conquest_scale)

    def test_fight_event_gives_kill_bonus(self):
        board = _board({"0,0": "p1", "1,0": "p2"})
        players = [_make_player("p1"), _make_player("p2")]
        event = MoveEvent(kind="fight", from_key="0,0", to_key="1,0", player_id="p1", units=4)
        r = compute_step_reward("p1", [event], board, board, players, players, winner_id=None)
        assert r == pytest.approx(DEFAULT_REWARD_CONFIG.kill_scale * 4)

    def test_enemy_events_ignored(self):
        """Events from other players should not affect our reward."""
        board = _board({"0,0": "p1", "1,0": "p2"})
        players = [_make_player("p1"), _make_player("p2")]
        event = MoveEvent(kind="conquer", from_key="1,0", to_key="2,0", player_id="p2", units=3)
        r = compute_step_reward("p1", [event], board, board, players, players, winner_id=None)
        assert r == pytest.approx(0.0)

    def test_elimination_bonus_with_our_events(self):
        board = _board({"0,0": "p1"})
        before = [_make_player("p1"), _make_player("p2")]
        after  = [_make_player("p1"), _make_player("p2", eliminated=True)]
        event = MoveEvent(kind="conquer", from_key="0,0", to_key="1,0", player_id="p1", units=5)
        r = compute_step_reward("p1", [event], board, board, before, after, winner_id=None)
        # Should include elimination bonus + conquer scale
        assert r > DEFAULT_REWARD_CONFIG.conquest_scale

    def test_custom_config(self):
        config = RewardConfig(win_reward=10.0, loss_reward=-10.0)
        board = _board({"0,0": "p1"})
        players = [_make_player("p1"), _make_player("p2")]
        r = compute_step_reward("p1", [], board, board, players, players,
                                winner_id="p1", config=config)
        assert r == pytest.approx(10.0)
