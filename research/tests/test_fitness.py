"""Tests for evolutionary/fitness.py — FitnessResult and evaluate_weights."""

import pytest

from hexwar.agents.evolutionary.fitness import FitnessResult, evaluate_weights
from hexwar.agents.greedy_agent import DEFAULT_WEIGHTS, N_FEATURES


class TestFitnessResult:
    def test_fields(self):
        r = FitnessResult(win_rate=0.5, avg_turns=120.0, avg_tiles_held=30.0, score=0.42)
        assert r.win_rate == 0.5
        assert r.avg_turns == 120.0
        assert r.avg_tiles_held == 30.0
        assert r.score == 0.42

    def test_score_formula(self):
        """Score = win_weight*win_rate - turns_weight*avg_turns + tiles_weight*avg_tiles."""
        win_weight, turns_weight, tiles_weight = 1.0, 0.001, 0.002
        win_rate, avg_turns, avg_tiles = 0.3, 100.0, 25.0
        expected = win_weight * win_rate - turns_weight * avg_turns + tiles_weight * avg_tiles
        r = FitnessResult(
            win_rate=win_rate,
            avg_turns=avg_turns,
            avg_tiles_held=avg_tiles,
            score=expected,
        )
        assert abs(r.score - expected) < 1e-9


# Run evaluate_weights once for the whole session; all output tests share the result.
@pytest.fixture(scope="session")
def eval_result():
    return evaluate_weights(
        list(DEFAULT_WEIGHTS) + [1.0],
        n_games=6,
        max_turns=30,
        win_weight=1.0,
        turns_weight=0.001,
        tiles_weight=0.002,
    )


class TestEvaluateWeights:
    def test_returns_fitness_result(self, eval_result):
        assert isinstance(eval_result, FitnessResult)

    def test_win_rate_in_range(self, eval_result):
        assert 0.0 <= eval_result.win_rate <= 1.0

    def test_avg_turns_positive(self, eval_result):
        assert eval_result.avg_turns >= 1.0

    def test_avg_tiles_nonnegative(self, eval_result):
        assert eval_result.avg_tiles_held >= 0.0

    def test_score_consistent_with_formula(self, eval_result):
        """Score must equal the weighted combination of the other fields."""
        expected = 1.0 * eval_result.win_rate - 0.001 * eval_result.avg_turns + 0.002 * eval_result.avg_tiles_held
        assert abs(eval_result.score - expected) < 1e-9

    def test_11dim_weights_accepted(self):
        """11-dim weights (no send_fraction) are accepted without error."""
        r = evaluate_weights(list(DEFAULT_WEIGHTS), n_games=2, max_turns=20)
        assert isinstance(r, FitnessResult)

    def test_seed_offset_changes_results(self):
        """Different seed offsets produce valid (potentially different) results."""
        weights = list(DEFAULT_WEIGHTS) + [1.0]
        r1 = evaluate_weights(weights, n_games=2, max_turns=20, seed_offset=0)
        r2 = evaluate_weights(weights, n_games=2, max_turns=20, seed_offset=999)
        assert isinstance(r1, FitnessResult)
        assert isinstance(r2, FitnessResult)
