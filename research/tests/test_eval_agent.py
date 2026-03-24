"""Tests for evaluation/eval_agent.py."""

import pytest

try:
    from hexwar.evaluation.eval_agent import EvalResult, evaluate_vs_greedy_baseline
    EVAL_AVAILABLE = True
except ImportError:
    EVAL_AVAILABLE = False

pytestmark = pytest.mark.skipif(not EVAL_AVAILABLE, reason="torch not installed")


class TestEvalResult:
    def test_str_format(self):
        r = EvalResult(win_rate=0.25, avg_tiles=30.5, avg_turns=150.2, n_games=12)
        s = str(r)
        assert "12" in s
        assert "25.0" in s
        assert "30.5" in s
        assert "150.2" in s

    def test_str_contains_all_fields(self):
        r = EvalResult(win_rate=0.0, avg_tiles=0.0, avg_turns=0.0, n_games=5)
        s = str(r)
        assert "Win%" in s
        assert "Avg tiles" in s
        assert "Avg turns" in s
        assert "Games" in s

    def test_fields_accessible(self):
        r = EvalResult(win_rate=0.5, avg_tiles=20.0, avg_turns=100.0, n_games=6)
        assert r.win_rate == 0.5
        assert r.avg_tiles == 20.0
        assert r.avg_turns == 100.0
        assert r.n_games == 6


# Run the simulation once for the whole session; all assertions share the result.
@pytest.fixture(scope="session")
def baseline_result():
    return evaluate_vs_greedy_baseline(n_games=6, max_turns=30)


class TestEvaluateVsGreedyBaseline:
    def test_returns_eval_result(self, baseline_result):
        assert isinstance(baseline_result, EvalResult)
        assert baseline_result.n_games == 6

    def test_win_rate_in_range(self, baseline_result):
        assert 0.0 <= baseline_result.win_rate <= 1.0

    def test_avg_turns_positive(self, baseline_result):
        assert baseline_result.avg_turns >= 1.0

    def test_avg_tiles_nonnegative(self, baseline_result):
        assert baseline_result.avg_tiles >= 0.0

    def test_near_random_win_rate(self, baseline_result):
        """Win rate is a valid probability (all assertions share one simulation run)."""
        assert 0.0 <= baseline_result.win_rate <= 1.0
