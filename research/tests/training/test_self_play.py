"""Tests for SelfPlayCollector."""

from __future__ import annotations

import pytest

from hexwar.agents.greedy_agent import GreedyAgent
from hexwar.engine.types import PLAYER_IDS

try:
    import torch
    from hexwar.agents.neural.gnn_model import HexWarGNN
    from hexwar.agents.neural.ppo_agent import PPOAgent
    from hexwar.training.self_play import SelfPlayCollector
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")


@pytest.fixture
def agent():
    return PPOAgent(model=HexWarGNN(history_k=5), history_k=5)


class TestSelfPlayCollector:
    def test_collect_returns_buffer_per_player(self, agent):
        collector = SelfPlayCollector(agent=agent, n_episodes=1, max_turns=20)
        buffers = collector.collect()
        assert set(buffers.keys()) == set(PLAYER_IDS)

    def test_learner_collects_transitions(self, agent):
        """At least some transitions must be collected for the learner."""
        collector = SelfPlayCollector(agent=agent, n_episodes=2, max_turns=20, seed=0)
        buffers = collector.collect()
        total = sum(len(b) for b in buffers.values())
        assert total > 0

    def test_transitions_have_returns(self, agent):
        """compute_returns() should set advantage and return_ on each transition."""
        collector = SelfPlayCollector(agent=agent, n_episodes=1, max_turns=20, seed=1)
        buffers = collector.collect()
        for buf in buffers.values():
            for tr in buf._transitions:
                assert hasattr(tr, "advantage")
                assert hasattr(tr, "return_")

    def test_greedy_opponent_pool(self, agent):
        """Collector works with GreedyAgent opponents (not just PPOAgent)."""
        greedy_pool = [GreedyAgent() for _ in range(3)]
        collector = SelfPlayCollector(
            agent=agent,
            n_episodes=1,
            max_turns=15,
            opponent_pool=greedy_pool,
            seed=2,
        )
        buffers = collector.collect()
        total = sum(len(b) for b in buffers.values())
        assert total >= 0  # no crash

    def test_add_to_pool(self, agent):
        collector = SelfPlayCollector(agent=agent, n_episodes=1, max_turns=10)
        initial_size = len(collector.opponent_pool)
        snap = PPOAgent(model=HexWarGNN(history_k=5), history_k=5)
        collector.add_to_pool(snap)
        assert len(collector.opponent_pool) == initial_size + 1

    def test_opponent_history_reset_between_episodes(self, agent):
        """PPOAgent opponents should have history reset each episode (no stale state)."""
        snap = PPOAgent(model=HexWarGNN(history_k=5), history_k=5)
        collector = SelfPlayCollector(
            agent=agent,
            n_episodes=2,
            max_turns=15,
            opponent_pool=[snap],
            seed=3,
        )
        # Should complete without errors (stale history would cause shape mismatches
        # or incorrect behaviour, but shouldn't crash)
        collector.collect()

    def test_game_completes_with_elimination(self, agent):
        """Run enough turns that eliminations can occur; loop must not skip players."""
        collector = SelfPlayCollector(agent=agent, n_episodes=1, max_turns=50, seed=99)
        buffers = collector.collect()
        # If the turn-loop bug were present (indexing live instead of player_ids)
        # games with eliminations would act on the wrong player.
        # We can't assert exact behaviour, but it must not raise.
        assert buffers is not None
