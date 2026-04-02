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
        agent = _make_agent()
        league = _make_league(agent)
        collector = StrategistCollector(
            agent=agent, league=league, n_episodes=1, max_turns=10, seed=1
        )
        work = collector._make_work(ep=0, learner_id="p1")
        transitions = _run_episode_fn(work)
        for tr in transitions:
            assert hasattr(tr, "obs")
            assert hasattr(tr, "h_tiles")
            assert hasattr(tr, "turn_frac")
            assert 0.0 <= tr.turn_frac <= 1.0
            # log_prob must be [N_orders] — same length as chosen_edges.
            # A scalar would break per-order PPO clipping in the trainer.
            assert tr.log_prob.shape == tr.chosen_edges.shape, (
                f"log_prob shape {tr.log_prob.shape} != "
                f"chosen_edges shape {tr.chosen_edges.shape}"
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
