"""
Tests for Behavioural Cloning warm-start.

Covers:
  - collect_bc_data: returns valid BCExample objects
  - bc_train max_examples: caps the dataset when too large, skips when small
  - bc_train: modifies model weights
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from hexwar.agents.neural.strategist_agent import StrategistAgent
from hexwar.agents.neural.strategist_model import StrategistGNN
from hexwar.training.behavioural_cloning import BCExample, bc_train, collect_bc_data


def _make_agent(hidden_dim: int = 32) -> StrategistAgent:
    model = StrategistGNN(hidden_dim=hidden_dim, n_layers=2, n_heads=2)
    return StrategistAgent(model=model, max_turns=20, deterministic=False)


# ---------------------------------------------------------------------------
# collect_bc_data
# ---------------------------------------------------------------------------

class TestCollectBCData:
    def test_returns_list_of_bc_examples(self):
        examples = collect_bc_data(n_games=2, max_turns=10, seed=0)
        assert isinstance(examples, list)
        assert len(examples) > 0
        assert all(isinstance(e, BCExample) for e in examples)

    def test_target_edge_within_graph(self):
        examples = collect_bc_data(n_games=2, max_turns=10, seed=1)
        for ex in examples:
            n_edges = ex.obs.edge_index.shape[1]
            assert 0 <= ex.target_edge < n_edges, (
                f"target_edge {ex.target_edge} out of range [0, {n_edges})"
            )

    def test_target_frac_in_unit_interval(self):
        examples = collect_bc_data(n_games=2, max_turns=10, seed=2)
        for ex in examples:
            assert 0.0 < ex.target_frac <= 1.0, (
                f"target_frac {ex.target_frac} not in (0, 1]"
            )

    def test_acting_mask_has_acting_tiles(self):
        examples = collect_bc_data(n_games=2, max_turns=10, seed=3)
        for ex in examples:
            assert ex.acting_mask.any(), "acting_mask must have at least one True tile"

    def test_more_games_produce_more_examples(self):
        ex2 = collect_bc_data(n_games=2, max_turns=15, seed=0)
        ex4 = collect_bc_data(n_games=4, max_turns=15, seed=0)
        assert len(ex4) > len(ex2)


# ---------------------------------------------------------------------------
# bc_train — max_examples cap
# ---------------------------------------------------------------------------

class TestBcTrainMaxExamples:
    def test_max_examples_samples_down(self):
        """When collected > max_examples, the dataset is randomly sampled down."""
        import random
        import unittest.mock as mock

        cap = 30
        sampled_ks: list[int] = []
        original = random.Random.sample

        def spy_sample(self_, population, k):
            sampled_ks.append(k)
            return original(self_, population, k)

        with mock.patch.object(random.Random, "sample", spy_sample):
            agent = _make_agent()
            # 5 games × 15 turns easily produces > 30 examples
            bc_train(agent, n_games=5, max_turns=15, n_epochs=1,
                     max_examples=cap, seed=0, verbose=False)

        assert len(sampled_ks) == 1, "sample() should be called exactly once"
        assert sampled_ks[0] == cap

    def test_no_sampling_when_max_examples_is_none(self):
        """max_examples=None disables the cap entirely."""
        import random
        import unittest.mock as mock

        sampled_ks: list[int] = []
        original = random.Random.sample

        def spy_sample(self_, population, k):
            sampled_ks.append(k)
            return original(self_, population, k)

        with mock.patch.object(random.Random, "sample", spy_sample):
            agent = _make_agent()
            bc_train(agent, n_games=2, max_turns=10, n_epochs=1,
                     max_examples=None, seed=0, verbose=False)

        assert len(sampled_ks) == 0, "sample() must not be called when max_examples=None"

    def test_no_sampling_when_below_cap(self):
        """When collected examples <= max_examples, no sampling occurs."""
        import random
        import unittest.mock as mock

        sampled_ks: list[int] = []
        original = random.Random.sample

        def spy_sample(self_, population, k):
            sampled_ks.append(k)
            return original(self_, population, k)

        with mock.patch.object(random.Random, "sample", spy_sample):
            agent = _make_agent()
            # cap is huge — collection will never reach it
            bc_train(agent, n_games=1, max_turns=5, n_epochs=1,
                     max_examples=1_000_000, seed=0, verbose=False)

        assert len(sampled_ks) == 0, "sample() must not be called when below cap"


# ---------------------------------------------------------------------------
# bc_train — weight update
# ---------------------------------------------------------------------------

class TestBcTrainWeightUpdate:
    def test_weights_change_after_bc(self):
        """BC training must update at least some model parameters."""
        agent = _make_agent()
        before = {k: v.clone() for k, v in agent.model.state_dict().items()}

        bc_train(agent, n_games=2, max_turns=10, n_epochs=2,
                 max_examples=100, seed=0, verbose=False)

        after = agent.model.state_dict()
        changed = any(not torch.equal(before[k], after[k]) for k in before)
        assert changed, "BC training should modify model weights"

    def test_model_in_eval_mode_after_bc(self):
        """bc_train must restore model to eval mode on exit."""
        agent = _make_agent()
        bc_train(agent, n_games=2, max_turns=10, n_epochs=1,
                 max_examples=50, seed=0, verbose=False)
        assert not agent.model.training, "model must be in eval mode after bc_train"
