"""
Smoke tests for StrategistTrainer.

Key invariants verified:
  - Per-order PPO clipping keeps loss bounded (no joint-ratio explosions).
  - log_prob is stored as [N_orders] not a scalar sum.
  - update() returns expected metric keys.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from hexwar.agents.greedy_agent import DEFAULT_WEIGHTS, GreedyAgent
from hexwar.agents.neural.strategist_agent import StrategistAgent
from hexwar.agents.neural.strategist_model import StrategistGNN
from hexwar.training.league import League, LeagueConfig
from hexwar.training.strategist_collect import StrategistCollector
from hexwar.training.strategist_train import StrategistTrainer


def _make_agent(hidden_dim: int = 32) -> StrategistAgent:
    model = StrategistGNN(hidden_dim=hidden_dim, n_layers=2, n_heads=2)
    return StrategistAgent(model=model, max_turns=30, deterministic=False)


def _make_league(agent: StrategistAgent) -> League:
    cfg = LeagueConfig(self_play_prob=0.8, snapshot_prob=0.1, greedy_prob=0.1)
    league = League(learner=agent, config=cfg, seed=0)
    league.add_greedy(GreedyAgent(weights=list(DEFAULT_WEIGHTS)))
    return league


def _collect(agent: StrategistAgent, league: League, seed: int = 0) -> object:
    collector = StrategistCollector(
        agent=agent, league=league, n_episodes=4, max_turns=20, seed=seed
    )
    return collector.collect()


# ---------------------------------------------------------------------------
# Metric keys
# ---------------------------------------------------------------------------

class TestStrategistTrainerMetrics:
    def test_update_returns_dict(self):
        agent = _make_agent()
        buf = _collect(agent, _make_league(agent))
        if len(buf) == 0:
            pytest.skip("No learner transitions collected")
        metrics = StrategistTrainer(agent=agent).update(buf)
        assert isinstance(metrics, dict)

    def test_update_returns_all_component_keys(self):
        agent = _make_agent()
        buf = _collect(agent, _make_league(agent), seed=1)
        if len(buf) == 0:
            pytest.skip("No learner transitions collected")
        metrics = StrategistTrainer(agent=agent).update(buf)
        for key in ("loss", "pg_loss", "vf_loss", "entropy"):
            assert key in metrics, f"Missing metric key: {key}"

    def test_empty_buffer_returns_empty_dict(self):
        from hexwar.training.strategist_collect import StrategistRolloutBuffer
        agent = _make_agent()
        buf = StrategistRolloutBuffer()
        metrics = StrategistTrainer(agent=agent).update(buf)
        assert metrics == {}


# ---------------------------------------------------------------------------
# Per-order PPO clipping — the core stability fix
# ---------------------------------------------------------------------------

class TestPerOrderClipping:
    def test_loss_stays_bounded_over_multiple_updates(self):
        """
        Joint log-prob clipping (the old bug) produced ratios like 1.2^10 ≈ 6x
        for a 10-order turn, causing catastrophic loss spikes (observed: 146,005
        and 2,972,500,966 in production logs).

        Per-order clipping bounds each order's ratio independently to [0.8, 1.2],
        so the loss must stay well below 1,000 throughout.
        """
        agent = _make_agent()
        league = _make_league(agent)
        trainer = StrategistTrainer(agent=agent)

        for i in range(5):
            buf = _collect(agent, league, seed=i * 10)
            if len(buf) == 0:
                continue
            metrics = trainer.update(buf)
            loss = metrics.get("loss", 0.0)
            assert abs(loss) < 1_000, (
                f"Loss explosion at iter {i}: {loss:.1f}. "
                "Per-order PPO clipping should keep this well below 1,000."
            )

    def test_log_prob_is_per_order_not_scalar(self):
        """
        Each transition's log_prob must have the same shape as chosen_edges
        ([N_orders]), not be a scalar sum.  A scalar breaks PPO clipping
        because exp(sum_N lp_new - sum_N lp_old) is the joint ratio, not
        the per-order ratio.
        """
        agent = _make_agent()
        buf = _collect(agent, _make_league(agent), seed=7)
        for tr in buf._transitions:
            assert tr.log_prob.shape == tr.chosen_edges.shape, (
                f"log_prob shape {tr.log_prob.shape} must equal "
                f"chosen_edges shape {tr.chosen_edges.shape}. "
                "Scalar log_prob indicates the old joint-sum bug is back."
            )

    def test_log_prob_is_1d_tensor(self):
        """log_prob must be a 1-D tensor, never a 0-D scalar."""
        agent = _make_agent()
        buf = _collect(agent, _make_league(agent), seed=3)
        for tr in buf._transitions:
            assert tr.log_prob.dim() == 1, (
                f"log_prob must be 1-D, got {tr.log_prob.dim()}-D. "
                "A 0-D tensor is the scalar sum — PPO clipping will not work."
            )

    def test_pg_loss_sign_is_valid(self):
        """
        pg_loss should be negative (we negate the objective) or close to zero,
        never a large positive number that would indicate an explosion.
        """
        agent = _make_agent()
        buf = _collect(agent, _make_league(agent), seed=5)
        if len(buf) == 0:
            pytest.skip("No learner transitions collected")
        metrics = StrategistTrainer(agent=agent).update(buf)
        pg = metrics.get("pg_loss", 0.0)
        assert abs(pg) < 100, f"pg_loss magnitude {pg:.2f} is suspiciously large"


# ---------------------------------------------------------------------------
# Checkpoint persistence
# ---------------------------------------------------------------------------

class TestCheckpointFormats:
    def test_trainer_checkpoint_roundtrip(self, tmp_path):
        """save_checkpoint / load_checkpoint preserves model weights and step."""
        agent = _make_agent()
        trainer = StrategistTrainer(agent=agent)
        trainer._step = 7

        path = tmp_path / "ckpt.pt"
        trainer.save_checkpoint(path)

        agent2 = _make_agent()
        trainer2 = StrategistTrainer(agent=agent2)
        trainer2.load_checkpoint(path)

        assert trainer2._step == 7
        for k in agent.model.state_dict():
            assert torch.allclose(
                agent.model.state_dict()[k].float(),
                agent2.model.state_dict()[k].float(),
            ), f"Weight mismatch after roundtrip: {k}"

    def test_weights_format_is_plain_state_dict(self, tmp_path):
        """agent.save() writes a plain state_dict — not the trainer checkpoint format."""
        agent = _make_agent()
        path = tmp_path / "bc.pt"
        agent.save(path)

        raw = torch.load(path, map_location="cpu")
        # Plain state_dict: keys are parameter names, values are tensors
        assert isinstance(raw, dict)
        assert "model" not in raw, (
            "agent.save() must write a plain state_dict, not a trainer checkpoint. "
            "If this fails, load_checkpoint() and agent.load() are now compatible "
            "and the --weights / --resume distinction is unnecessary."
        )
        # All values must be tensors (parameter weights)
        assert all(isinstance(v, torch.Tensor) for v in raw.values())

    def test_weights_load_via_agent_load(self, tmp_path):
        """Weights saved by agent.save() can be reloaded by agent.load()."""
        agent = _make_agent()
        path = tmp_path / "bc.pt"
        agent.save(path)

        import copy
        agent2 = StrategistAgent(
            model=copy.deepcopy(agent.model), deterministic=True
        )
        # Perturb weights so we can verify load actually changes them
        with torch.no_grad():
            for p in agent2.model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        agent2.load(path)

        for k in agent.model.state_dict():
            assert torch.allclose(
                agent.model.state_dict()[k].float(),
                agent2.model.state_dict()[k].float(),
            ), f"Weight mismatch after agent.load(): {k}"
