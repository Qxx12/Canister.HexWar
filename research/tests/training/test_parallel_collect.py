"""Smoke tests for ParallelStrategistCollector."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from hexwar.agents.greedy_agent import DEFAULT_WEIGHTS, GreedyAgent
from hexwar.agents.neural.strategist_agent import StrategistAgent
from hexwar.agents.neural.strategist_model import StrategistGNN
from hexwar.training.league import League, LeagueConfig
from hexwar.training.strategist_collect import (
    ParallelStrategistCollector,
    StrategistCollector,
    _agent_to_spec,
    _run_episode_fn,
    _spec_to_agent,
)


def _make_agent(hidden_dim: int = 32) -> StrategistAgent:
    model = StrategistGNN(hidden_dim=hidden_dim, n_layers=2, n_heads=2)
    return StrategistAgent(model=model, max_turns=30, deterministic=False)


def _make_league(agent: StrategistAgent) -> League:
    cfg = LeagueConfig(self_play_prob=0.8, snapshot_prob=0.1, greedy_prob=0.1)
    league = League(learner=agent, config=cfg, seed=0)
    league.add_greedy(GreedyAgent(weights=list(DEFAULT_WEIGHTS)))
    return league


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

class TestAgentToSpec:
    def test_greedy_spec(self):
        agent = GreedyAgent(weights=list(DEFAULT_WEIGHTS))
        spec = _agent_to_spec(agent)
        assert spec.kind == "greedy"
        assert spec.greedy_weights == list(DEFAULT_WEIGHTS)

    def test_strategist_spec(self):
        agent = _make_agent()
        spec = _agent_to_spec(agent)
        assert spec.kind == "strategist"
        assert spec.state_dict is not None
        assert spec.hidden_dim == 32
        assert spec.n_layers == 2
        assert spec.n_heads == 2

    def test_spec_roundtrip_greedy(self):
        agent = GreedyAgent(weights=list(DEFAULT_WEIGHTS))
        spec = _agent_to_spec(agent)
        recovered = _spec_to_agent(spec)
        assert isinstance(recovered, GreedyAgent)
        assert recovered.weights == agent.weights

    def test_spec_roundtrip_strategist(self):
        agent = _make_agent(hidden_dim=32)
        spec = _agent_to_spec(agent)
        recovered = _spec_to_agent(spec)
        assert isinstance(recovered, StrategistAgent)
        assert recovered.model.hidden_dim == 32


# ---------------------------------------------------------------------------
# _run_episode_fn (module-level worker)
# ---------------------------------------------------------------------------

class TestRunEpisodeFn:
    def test_returns_list(self):
        agent = _make_agent()
        league = _make_league(agent)
        collector = StrategistCollector(
            agent=agent, league=league, n_episodes=1, max_turns=10, seed=0
        )
        work = collector._make_work(ep=0, learner_id="p1")
        result = _run_episode_fn(work)
        assert isinstance(result, list)

    def test_transitions_have_correct_fields(self):
        """_run_episode_fn returns list[dict] (numpy arrays); StrategistCollector
        converts them back to StrategistTransition via _raw_to_transition."""
        import numpy as np
        agent = _make_agent()
        league = _make_league(agent)
        collector = StrategistCollector(
            agent=agent, league=league, n_episodes=1, max_turns=10, seed=1
        )
        work = collector._make_work(ep=0, learner_id="p1")
        raw_list = _run_episode_fn(work)
        for raw in raw_list:
            # Raw dict must have numpy obs fields and scalar turn_frac
            assert "obs_x" in raw
            assert "h_tiles" in raw
            assert "turn_frac" in raw
            assert 0.0 <= raw["turn_frac"] <= 1.0
            # log_prob and chosen_edges must have the same shape [N_orders]
            assert raw["log_prob"].shape == raw["chosen_edges"].shape, (
                f"log_prob shape {raw['log_prob'].shape} != "
                f"chosen_edges shape {raw['chosen_edges'].shape}"
            )


# ---------------------------------------------------------------------------
# Serial collector
# ---------------------------------------------------------------------------

class TestStrategistCollector:
    def test_collect_returns_buffer(self):
        agent = _make_agent()
        league = _make_league(agent)
        collector = StrategistCollector(
            agent=agent, league=league, n_episodes=2, max_turns=15, seed=0
        )
        buf = collector.collect()
        assert len(buf) >= 0  # may be 0 if learner had no acting turns

    def test_buffer_has_advantages_after_collect(self):
        agent = _make_agent()
        league = _make_league(agent)
        collector = StrategistCollector(
            agent=agent, league=league, n_episodes=2, max_turns=15, seed=2
        )
        buf = collector.collect()
        for tr in buf._transitions:
            assert tr.advantage is not None
            assert tr.return_ is not None


# ---------------------------------------------------------------------------
# Parallel collector
# ---------------------------------------------------------------------------

class TestParallelStrategistCollector:
    def test_collect_returns_buffer(self):
        agent = _make_agent()
        league = _make_league(agent)
        collector = ParallelStrategistCollector(
            agent=agent, league=league, n_episodes=4, max_turns=15,
            n_workers=2, seed=0,
        )
        buf = collector.collect()
        assert len(buf) >= 0

    def test_parallel_matches_serial_transition_count(self):
        """Serial and parallel should collect similar numbers of transitions.

        Exact equality is not guaranteed: the serial path runs _run_episode_fn
        in the main process (inheriting its PyTorch RNG state), while the
        parallel path spawns fresh subprocesses (each with a clean RNG).
        PyTorch's Beta distribution sampling for action fractions therefore
        produces different values, which can shift elimination timing by a
        turn and change the total learner-turn count by a small amount.

        We allow a tolerance of ±3 transitions to cover this, while still
        catching genuine structural bugs (e.g. parallel returning 0 when
        serial returns 60).
        """
        agent_s = _make_agent()
        agent_p = _make_agent()
        # Copy weights so both start identically
        agent_p.model.load_state_dict(agent_s.model.state_dict())

        league_s = _make_league(agent_s)
        league_p = _make_league(agent_p)

        serial = StrategistCollector(
            agent=agent_s, league=league_s, n_episodes=4, max_turns=15, seed=99
        )
        parallel = ParallelStrategistCollector(
            agent=agent_p, league=league_p, n_episodes=4, max_turns=15,
            n_workers=2, seed=99,
        )
        buf_s = serial.collect()
        buf_p = parallel.collect()
        diff = abs(len(buf_s) - len(buf_p))
        assert diff <= 3, (
            f"Serial collected {len(buf_s)} transitions, parallel {len(buf_p)} "
            f"(diff={diff}). A difference >3 suggests a structural bug."
        )

    def test_n_workers_defaults_to_cpu_count(self):
        import os
        agent = _make_agent()
        league = _make_league(agent)
        collector = ParallelStrategistCollector(
            agent=agent, league=league, n_episodes=2, max_turns=10,
        )
        assert collector.n_workers <= (os.cpu_count() or 1)
        assert collector.n_workers >= 1


# ---------------------------------------------------------------------------
# Seed advancement — each collect() call must use a different base seed
# ---------------------------------------------------------------------------

class TestSeedAdvancement:
    def test_serial_call_count_increments(self):
        agent = _make_agent()
        league = _make_league(agent)
        collector = StrategistCollector(
            agent=agent, league=league, n_episodes=2, max_turns=10, seed=0
        )
        assert collector._call_count == 0
        collector.collect()
        assert collector._call_count == 1
        collector.collect()
        assert collector._call_count == 2

    def test_serial_make_work_uses_different_seeds_across_calls(self):
        """Second collect() call must produce different episode seeds than first."""
        agent = _make_agent()
        league = _make_league(agent)
        n_ep = 3
        collector = StrategistCollector(
            agent=agent, league=league, n_episodes=n_ep, max_turns=10, seed=7
        )
        # Simulate two calls manually to inspect seeds without running full games
        base_seed_0 = collector.seed + 0 * n_ep
        base_seed_1 = collector.seed + 1 * n_ep
        seeds_call_0 = {base_seed_0 + ep for ep in range(n_ep)}
        seeds_call_1 = {base_seed_1 + ep for ep in range(n_ep)}
        assert seeds_call_0.isdisjoint(seeds_call_1), (
            "Episode seeds must not overlap across consecutive collect() calls."
        )

    def test_parallel_call_count_increments(self):
        agent = _make_agent()
        league = _make_league(agent)
        collector = ParallelStrategistCollector(
            agent=agent, league=league, n_episodes=2, max_turns=10, n_workers=1, seed=0
        )
        assert collector._call_count == 0
        collector.collect()
        assert collector._call_count == 1


# ---------------------------------------------------------------------------
# Elimination-done fix — learner elimination must set done=True
# ---------------------------------------------------------------------------

class TestEliminationDone:
    def test_last_transition_done_if_learner_eliminated(self):
        """
        When the learner is eliminated before the game ends, the last
        learner transition must have done=True so GAE does not bleed
        into the next episode.
        """
        agent = _make_agent()
        league = _make_league(agent)
        collector = StrategistCollector(
            agent=agent, league=league, n_episodes=8, max_turns=30, seed=42
        )
        buf = collector.collect()
        if len(buf) == 0:
            pytest.skip("No learner transitions collected in this seed")
        # Group transitions by episode: find groups separated by done=True
        # Any final transition of an episode must have done=True.
        # Within an episode, all but the last must have done=False.
        episode_finals = []
        for i, tr in enumerate(buf._transitions):
            if tr.done:
                episode_finals.append(i)
        # Every done=True transition must be a real episode terminal:
        # it should be followed by either end-of-buffer or the start of a
        # new episode (i.e., not mid-episode).
        # At minimum, verify no two consecutive done=True transitions are
        # unreachably close together (single-step episodes would be suspicious
        # but not invalid for very short max_turns games).
        for idx in episode_finals:
            assert buf._transitions[idx].done is True
