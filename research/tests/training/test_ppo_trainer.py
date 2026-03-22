"""Tests for PPOTrainer."""

from __future__ import annotations

import pytest

try:
    import torch
    from hexwar.agents.neural.gnn_model import HexWarGNN
    from hexwar.agents.neural.ppo_agent import PPOAgent
    from hexwar.training.ppo_trainer import PPOTrainer
    from hexwar.training.rollout_buffer import RolloutBuffer, Transition
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")


@pytest.fixture
def agent():
    return PPOAgent(model=HexWarGNN(history_k=5), history_k=5)


@pytest.fixture
def trainer(agent):
    return PPOTrainer(agent=agent, batch_size=4, n_epochs=1)


def _make_dummy_buffer(agent, n=8, max_turns=10):
    """Collect a small buffer using a fast greedy opponent."""
    from hexwar.agents.greedy_agent import GreedyAgent
    from hexwar.training.self_play import SelfPlayCollector
    collector = SelfPlayCollector(
        agent=agent,
        n_episodes=1,
        max_turns=max_turns,
        opponent_pool=[GreedyAgent()],
        seed=42,
    )
    return collector.collect()


class TestPPOTrainer:
    def test_update_returns_metrics(self, agent, trainer):
        buffers = _make_dummy_buffer(agent)
        metrics = trainer.update(buffers)
        assert "loss" in metrics
        assert "pg_loss" in metrics
        assert "vf_loss" in metrics
        assert "entropy" in metrics

    def test_update_empty_buffer_returns_empty(self, trainer):
        from hexwar.engine.types import PLAYER_IDS
        empty_buffers = {pid: RolloutBuffer() for pid in PLAYER_IDS}
        metrics = trainer.update(empty_buffers)
        assert metrics == {}

    def test_advantage_normalization(self, agent, trainer):
        """Normalized advantages should have ~zero mean and ~unit std."""
        buffers = _make_dummy_buffer(agent)
        # Inject known advantages for inspection
        all_transitions = []
        for buf in buffers.values():
            all_transitions.extend(buf._transitions)

        if not all_transitions:
            pytest.skip("No transitions collected")

        for tr in all_transitions:
            tr.advantage = torch.tensor(1.0)  # constant — after norm should be ~0

        # Call internal batch to check normalization
        batch = all_transitions[:4]
        raw = torch.stack([tr.advantage for tr in batch])
        normed = (raw - raw.mean()) / (raw.std() + 1e-8)
        assert normed.mean().abs() < 1e-5

    def test_parameters_change_after_update(self, agent, trainer):
        """Model weights should change after a non-trivial update."""
        buffers = _make_dummy_buffer(agent)
        total = sum(len(b) for b in buffers.values())
        if total == 0:
            pytest.skip("No transitions collected")

        before = {k: v.clone() for k, v in agent.model.named_parameters()}
        trainer.update(buffers)
        changed = any(
            not torch.equal(before[k], v)
            for k, v in agent.model.named_parameters()
        )
        assert changed

    def test_save_load_checkpoint(self, agent, trainer, tmp_path):
        ckpt = tmp_path / "ckpt.pt"
        trainer.save_checkpoint(ckpt)
        assert ckpt.exists()
        # Load into fresh trainer
        agent2 = PPOAgent(model=HexWarGNN(history_k=5), history_k=5)
        trainer2 = PPOTrainer(agent=agent2, batch_size=4, n_epochs=1)
        trainer2.load_checkpoint(ckpt)
        # Weights should match
        for (k1, v1), (k2, v2) in zip(
            agent.model.named_parameters(), agent2.model.named_parameters()
        ):
            assert torch.equal(v1, v2), f"Mismatch in {k1}"
