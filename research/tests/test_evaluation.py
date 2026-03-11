"""Tests for the Elo rating system and tournament runner."""

import pytest

from hexwar.agents.random_agent import RandomAgent
from hexwar.evaluation.elo import EloSystem
from hexwar.evaluation.tournament import AgentStats, run_tournament


class TestEloSystem:
    def test_initial_rating(self):
        elo = EloSystem(initial_rating=1200)
        assert elo.ratings["new_player"] == pytest.approx(1200.0)

    def test_winner_gains_rating(self):
        elo = EloSystem()
        before = elo.ratings["a"]
        elo.update("a", "b", winner="a")
        assert elo.ratings["a"] > before

    def test_loser_loses_rating(self):
        elo = EloSystem()
        before = elo.ratings["b"]
        elo.update("a", "b", winner="a")
        assert elo.ratings["b"] < before

    def test_draw_moves_toward_equal(self):
        """Draw moves both players toward the same rating."""
        elo = EloSystem()
        elo.ratings["strong"] = 1500.0
        elo.ratings["weak"] = 900.0
        elo.update("strong", "weak", winner=None)
        # Strong should lose, weak should gain
        assert elo.ratings["strong"] < 1500.0
        assert elo.ratings["weak"] > 900.0

    def test_zero_sum(self):
        """Rating gains and losses cancel out in a 1v1."""
        elo = EloSystem()
        r_a_before = elo.ratings["a"]
        r_b_before = elo.ratings["b"]
        elo.update("a", "b", winner="a")
        delta_a = elo.ratings["a"] - r_a_before
        delta_b = elo.ratings["b"] - r_b_before
        assert delta_a + delta_b == pytest.approx(0.0, abs=1e-9)

    def test_game_counts_increment(self):
        elo = EloSystem()
        elo.update("a", "b", winner="a")
        assert elo.game_counts["a"] == 1
        assert elo.game_counts["b"] == 1

    def test_expected_score_equal_ratings(self):
        elo = EloSystem()
        assert elo.expected_score(1200, 1200) == pytest.approx(0.5)

    def test_expected_score_higher_wins_more(self):
        elo = EloSystem()
        assert elo.expected_score(1400, 1200) > 0.5

    def test_update_multiplayer_winner_gains(self):
        elo = EloSystem()
        players = ["p0", "p1", "p2", "p3", "p4", "p5"]
        before = elo.ratings["p0"]
        elo.update_multiplayer(players, winner_id="p0")
        assert elo.ratings["p0"] > before

    def test_update_multiplayer_no_winner(self):
        """With no winner, all ratings stay near initial (draws cancel out)."""
        elo = EloSystem(initial_rating=1200)
        players = ["p0", "p1", "p2", "p3", "p4", "p5"]
        elo.update_multiplayer(players, winner_id=None)
        # Symmetric game: all start equal, all draws → no net change
        for pid in players:
            assert elo.ratings[pid] == pytest.approx(1200.0, abs=1e-9)

    def test_print_rankings_runs(self, capsys):
        elo = EloSystem()
        elo.update("a", "b", winner="a")
        elo.print_rankings()
        out = capsys.readouterr().out
        assert "a" in out
        assert "b" in out


class TestTournament:
    def test_returns_stats_for_each_agent(self):
        agents = {"r1": RandomAgent(seed=0), "r2": RandomAgent(seed=1)}
        results = run_tournament(agents, n_games=2, max_turns=50, seed=42)
        assert set(results.keys()) == {"r1", "r2"}

    def test_game_counts_correct(self):
        agents = {"r1": RandomAgent(seed=0), "r2": RandomAgent(seed=1)}
        n_games = 4
        results = run_tournament(agents, n_games=n_games, max_turns=50, seed=0)
        # With 2 agents filling 6 slots, each agent occupies 3 slots per game.
        # games counter increments once per slot per game.
        for stats in results.values():
            assert stats.games == n_games * 3

    def test_win_rate_in_range(self):
        agents = {"r1": RandomAgent(seed=0), "r2": RandomAgent(seed=1)}
        results = run_tournament(agents, n_games=6, max_turns=100, seed=7)
        for stats in results.values():
            assert 0.0 <= stats.win_rate <= 1.0

    def test_avg_tiles_positive(self):
        agents = {"r1": RandomAgent(seed=0), "r2": RandomAgent(seed=1)}
        results = run_tournament(agents, n_games=2, max_turns=30, seed=3)
        for stats in results.values():
            assert stats.avg_tiles >= 0.0

    def test_agent_stats_properties(self):
        s = AgentStats(name="test", wins=3, games=10, total_turns=500, total_tiles=200)
        assert s.win_rate == pytest.approx(0.3)
        assert s.avg_turns == pytest.approx(50.0)
        assert s.avg_tiles == pytest.approx(20.0)

    def test_agent_stats_zero_games(self):
        s = AgentStats(name="test")
        assert s.win_rate == 0.0
        assert s.avg_turns == 0.0
        assert s.avg_tiles == 0.0
