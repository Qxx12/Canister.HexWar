"""Tests for the League opponent pool and sampling."""

import pytest

from hexwar.agents.base_agent import BaseAgent
from hexwar.engine.types import PLAYER_IDS


class _DummyAgent(BaseAgent):
    """Minimal agent that does nothing — for pool testing."""
    def __call__(self, board, player_id, players, stats):
        return {}

    def reset(self, initial_board=None):
        pass


def _make_league(n_greedy: int = 0, n_snapshots: int = 0):
    from hexwar.training.league import League, LeagueConfig

    learner = _DummyAgent()
    cfg = LeagueConfig(
        self_play_prob=0.5,
        snapshot_prob=0.3,
        greedy_prob=0.2,
        max_snapshots=10,
        snapshot_interval=5,
    )
    league = League(learner=learner, config=cfg, seed=0)
    for _ in range(n_greedy):
        league.add_greedy(_DummyAgent())
    for _ in range(n_snapshots):
        league._snapshots.append(_DummyAgent())
    return league


class TestLeaguePoolManagement:
    def test_initial_state_empty_pools(self):
        league = _make_league()
        s = league.stats()
        assert s["n_snapshots"] == 0
        assert s["n_greedy"] == 0

    def test_add_greedy_increases_pool(self):
        league = _make_league()
        league.add_greedy(_DummyAgent())
        assert league.stats()["n_greedy"] == 1

    def test_snapshot_increases_pool(self):
        league = _make_league()
        league.snapshot()
        assert league.stats()["n_snapshots"] == 1

    def test_snapshot_caps_at_max_snapshots(self):
        from hexwar.training.league import League, LeagueConfig
        cfg = LeagueConfig(max_snapshots=3)
        learner = _DummyAgent()
        league = League(learner=learner, config=cfg, seed=0)
        for _ in range(5):
            league.snapshot()
        assert len(league._snapshots) == 3

    def test_on_update_auto_snapshot(self):
        from hexwar.training.league import League, LeagueConfig
        cfg = LeagueConfig(snapshot_interval=3)
        league = League(learner=_DummyAgent(), config=cfg, seed=0)
        for _ in range(3):
            league.on_update()
        # After 3 calls, one auto-snapshot should have fired
        assert league.stats()["n_snapshots"] == 1

    def test_on_update_increments_counter(self):
        league = _make_league()
        for _ in range(7):
            league.on_update()
        assert league.stats()["updates"] == 7


class TestLeagueSampleOpponents:
    def test_learner_in_assigned_slot(self):
        league = _make_league()
        agents = league.sample_opponents("p1", PLAYER_IDS)
        assert agents["p1"] is league.learner

    def test_all_slots_filled(self):
        league = _make_league()
        agents = league.sample_opponents("p1", PLAYER_IDS)
        assert len(agents) == len(PLAYER_IDS)
        for pid in PLAYER_IDS:
            assert pid in agents

    def test_no_greedy_no_crash(self):
        """With empty greedy pool, all weight shifts to self-play."""
        league = _make_league(n_greedy=0, n_snapshots=0)
        agents = league.sample_opponents("p1", PLAYER_IDS)
        for pid, ag in agents.items():
            if pid != "p1":
                assert ag is league.learner

    def test_sampling_with_snapshots(self):
        """With snapshots available, some opponents may come from pool."""
        league = _make_league(n_snapshots=5)
        # Just verify it doesn't crash and returns right number of agents
        agents = league.sample_opponents("p2", PLAYER_IDS)
        assert len(agents) == len(PLAYER_IDS)

    def test_sampling_with_all_pools(self):
        league = _make_league(n_greedy=3, n_snapshots=3)
        agents = league.sample_opponents("p3", PLAYER_IDS)
        assert len(agents) == len(PLAYER_IDS)
        assert agents["p3"] is league.learner
