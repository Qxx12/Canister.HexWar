"""
League-based rollout collector for StrategistAgent.

Two collectors are provided:
  StrategistCollector         — serial (single process)
  ParallelStrategistCollector — parallel via ProcessPoolExecutor

Parallel design
---------------
The bottleneck in training is game simulation, not the model forward pass.
Each episode is fully independent (different board seed, different opponents),
so they can run in separate processes with zero synchronisation overhead.

Worker design:
  1. Main process pre-samples episode configs (learner_id, opponent draws).
  2. Each config is serialised into an _EpisodeWork dataclass containing:
       - Current learner weights (state_dict)
       - Serialised opponent specs (greedy weights or strategist state_dict)
  3. Workers receive _EpisodeWork, reconstruct agents locally, run the episode,
     and return raw StrategistTransition objects.
  4. Main process aggregates all transitions and computes GAE returns.

IMPORTANT: each worker calls torch.set_num_threads(1) to prevent thread-count
explosion (N workers × M PyTorch threads ≫ CPU cores causes severe contention).
"""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Beta as _Beta

if TYPE_CHECKING:
    from torch_geometric.data import Data

from ..agents.base_agent import BaseAgent
from ..agents.greedy_agent import GreedyAgent
from ..agents.neural.encoder import encode_board
from ..agents.neural.strategist_agent import StrategistAgent
from ..agents.neural.strategist_model import StrategistGNN
from ..engine.game_engine import HexWarEnv
from ..engine.turn_resolver import resolve_player_turn
from ..engine.types import PLAYER_IDS, MovementOrder, OrderMap
from ..engine.unit_generator import generate_units
from .league import League
from .reward import DEFAULT_REWARD_CONFIG, RewardConfig, compute_step_reward

# ---------------------------------------------------------------------------
# Transition + Buffer
# ---------------------------------------------------------------------------

@dataclass
class StrategistTransition:
    """Extended Transition that carries GRU state for re-evaluation."""
    obs: Data
    acting_mask: Tensor
    chosen_edges: Tensor
    chosen_fractions: Tensor
    log_prob: Tensor
    value: Tensor
    reward: float
    done: bool
    h_tiles: Tensor | None          # [N, hidden_dim] at collection time
    turn_frac: float                # normalised turn used at collection time
    advantage: Tensor | None = None
    return_: Tensor | None = None


class StrategistRolloutBuffer:
    """GAE rollout buffer for StrategistTransition objects."""

    def __init__(self, gamma: float = 0.99, lam: float = 0.95) -> None:
        self.gamma = gamma
        self.lam = lam
        self._transitions: list[StrategistTransition] = []

    def add(self, t: StrategistTransition) -> None:
        self._transitions.append(t)

    def __len__(self) -> int:
        return len(self._transitions)

    def clear(self) -> None:
        self._transitions.clear()

    def compute_returns(self) -> None:
        n = len(self._transitions)
        if n == 0:
            return
        gae = 0.0
        advantages = torch.zeros(n)
        for t in reversed(range(n)):
            tr = self._transitions[t]
            done = float(tr.done)
            value = tr.value.item()
            next_value = self._transitions[t + 1].value.item() if t + 1 < n else 0.0
            delta = tr.reward + self.gamma * next_value * (1 - done) - value
            gae = delta + self.gamma * self.lam * (1 - done) * gae
            advantages[t] = gae
        returns = advantages + torch.tensor([tr.value.item() for tr in self._transitions])
        for t, tr in enumerate(self._transitions):
            tr.advantage = advantages[t]
            tr.return_ = returns[t]

    def iter_minibatches(self, batch_size: int):
        indices = torch.randperm(len(self._transitions))
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            yield [self._transitions[i] for i in batch_idx]


# ---------------------------------------------------------------------------
# Serialisable agent specs (for subprocess workers)
# ---------------------------------------------------------------------------

@dataclass
class _OpponentSpec:
    """Fully picklable description of one agent used in a worker."""
    kind: str                           # "greedy" | "strategist"
    greedy_weights: list[float] | None  # greedy only
    state_dict: dict | None             # strategist only
    hidden_dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    max_turns: int = 300


@dataclass
class _EpisodeWork:
    """Everything a worker process needs to run one episode."""
    seed: int
    learner_id: str
    learner_spec: _OpponentSpec
    opponent_specs: dict[str, _OpponentSpec]  # pid → spec (all players)
    max_turns: int
    reward_gamma: float = 0.99
    reward_lam: float = 0.95
    reward_config: RewardConfig = field(default_factory=RewardConfig)


def _agent_to_spec(agent: BaseAgent, default_max_turns: int = 300) -> _OpponentSpec:
    """Convert any supported agent to a picklable _OpponentSpec.

    State-dict tensors are stored as numpy arrays so the spec is serialised by
    standard pickle without torch's mmap-based shared-memory mechanism.  With
    128+ work items in flight simultaneously, torch's file_system strategy
    creates tens of thousands of mmap files and hits vm.max_map_count.
    """
    if isinstance(agent, GreedyAgent):
        return _OpponentSpec(
            kind="greedy",
            greedy_weights=agent.weights,
            state_dict=None,
        )
    if isinstance(agent, StrategistAgent):
        model = agent.model
        return _OpponentSpec(
            kind="strategist",
            greedy_weights=None,
            state_dict={k: v.cpu().detach().numpy().copy()
                        for k, v in model.state_dict().items()},
            hidden_dim=model.hidden_dim,
            n_layers=len(model.conv_layers),
            n_heads=model.global_attn.num_heads,
            dropout=model.dropout,
            max_turns=getattr(agent, "max_turns", default_max_turns),
        )
    # Fallback: treat unknown agents as default greedy
    from ..agents.greedy_agent import DEFAULT_WEIGHTS
    return _OpponentSpec(kind="greedy", greedy_weights=list(DEFAULT_WEIGHTS), state_dict=None)


def _spec_to_agent(spec: _OpponentSpec) -> BaseAgent:
    """Reconstruct an agent from its picklable spec (runs inside worker)."""
    if spec.kind == "greedy":
        return GreedyAgent(weights=spec.greedy_weights)
    # strategist — state_dict values are numpy arrays; convert back to tensors
    model = StrategistGNN(
        hidden_dim=spec.hidden_dim,
        n_layers=spec.n_layers,
        n_heads=spec.n_heads,
        dropout=spec.dropout,
    )
    model.load_state_dict({k: torch.from_numpy(np.asarray(v))
                           for k, v in spec.state_dict.items()})
    model.eval()
    return StrategistAgent(model=model, max_turns=spec.max_turns, deterministic=False)


# ---------------------------------------------------------------------------
# Numpy ↔ StrategistTransition conversion helpers
#
# Workers return plain dicts of numpy arrays instead of StrategistTransition
# objects containing torch tensors.  Standard pickle serialises numpy arrays
# with zero mmap overhead, completely bypassing torch's shared-memory
# mechanism (file_system or file_descriptor).  The main process converts
# the dicts back to StrategistTransition after pool.map() returns.
# ---------------------------------------------------------------------------

def _transition_to_raw(tr: StrategistTransition) -> dict:
    """Flatten a StrategistTransition to a plain dict of numpy arrays + scalars."""
    obs = tr.obs
    return {
        "obs_x":         obs.x.numpy(),
        "obs_edge_index": obs.edge_index.numpy(),
        "obs_edge_attr": obs.edge_attr.numpy() if obs.edge_attr is not None else None,
        "obs_u":         obs.u.numpy()         if obs.u         is not None else None,
        "obs_node_keys": obs.node_keys,
        "acting_mask":   tr.acting_mask.numpy(),
        "chosen_edges":  tr.chosen_edges.numpy(),
        "chosen_fractions": tr.chosen_fractions.numpy(),
        "log_prob":      tr.log_prob.numpy(),
        "value":         float(tr.value),
        "reward":        tr.reward,
        "done":          tr.done,
        "h_tiles":       tr.h_tiles.numpy() if tr.h_tiles is not None else None,
        "turn_frac":     tr.turn_frac,
    }


def _raw_to_transition(d: dict) -> StrategistTransition:
    """Reconstruct a StrategistTransition from the dict produced by _transition_to_raw."""
    from torch_geometric.data import Data
    obs = Data(
        x          = torch.from_numpy(d["obs_x"]),
        edge_index = torch.from_numpy(d["obs_edge_index"]),
        edge_attr  = torch.from_numpy(d["obs_edge_attr"]) if d["obs_edge_attr"] is not None else None,
        u          = torch.from_numpy(d["obs_u"])         if d["obs_u"]         is not None else None,
    )
    obs.node_keys = d["obs_node_keys"]  # type: ignore[attr-defined]
    return StrategistTransition(
        obs               = obs,
        acting_mask       = torch.from_numpy(d["acting_mask"]),
        chosen_edges      = torch.from_numpy(d["chosen_edges"]),
        chosen_fractions  = torch.from_numpy(d["chosen_fractions"]),
        log_prob          = torch.from_numpy(d["log_prob"]),
        value             = torch.tensor(d["value"]),
        reward            = d["reward"],
        done              = d["done"],
        h_tiles           = torch.from_numpy(d["h_tiles"]) if d["h_tiles"] is not None else None,
        turn_frac         = d["turn_frac"],
    )


# ---------------------------------------------------------------------------
# Module-level episode worker (must be at module level for pickle / spawn)
# ---------------------------------------------------------------------------

def _run_episode_fn(work: _EpisodeWork) -> list[dict]:
    """
    Run one complete episode and return collected transitions as plain dicts.

    Returns list[dict] (numpy arrays + scalars) instead of
    list[StrategistTransition] so that standard pickle is used for IPC — no
    torch shared-memory (mmap) at all.  The main process converts the dicts
    back to StrategistTransition via _raw_to_transition().

    This function is called in a subprocess by ParallelStrategistCollector.
    It is also used by the serial StrategistCollector to avoid duplicating logic.

    Safety: torch.set_num_threads(1) prevents thread-count explosion when
    many workers run simultaneously.
    """
    torch.set_num_threads(1)

    # Reconstruct agents
    learner = _spec_to_agent(work.learner_spec)
    agents: dict[str, BaseAgent] = {}
    for pid in work.opponent_specs:
        agents[pid] = learner if pid == work.learner_id else _spec_to_agent(work.opponent_specs[pid])

    env = HexWarEnv(agents={}, seed=work.seed, max_turns=work.max_turns)

    for _pid, ag in agents.items():
        if hasattr(ag, "reset"):
            ag.reset(env.board)

    transitions: list[dict] = []
    turn_number = 0

    while env.phase != "end":

        # ── Unit generation ──────────────────────────────────────────────
        if env.phase == "generateUnits":
            generate_units(env.board, env.stats)
            turn_number += 1
            env.turn.turn_number += 1
            env.turn.active_ai_index = 0
            if env.turn.turn_number > work.max_turns:
                env._end_game(winner_id=None)
                break
            env.phase = "playerTurn"
            continue

        idx = env.turn.active_ai_index
        if idx >= len(env.player_ids):
            env.phase = "generateUnits"
            continue

        pid = env.player_ids[idx]
        env.turn.active_ai_index += 1

        player = next((p for p in env.players if p.id == pid), None)
        if player is None or player.is_eliminated:
            continue

        agent = agents[pid]
        board_before = deepcopy(env.board)
        players_before = deepcopy(env.players)

        orders: OrderMap = {}
        value_est = torch.tensor(0.0)
        chosen_edges = torch.zeros(0, dtype=torch.long)
        chosen_fracs = torch.zeros(0)
        log_p = torch.tensor(0.0)
        acting_mask_t = torch.zeros(0, dtype=torch.bool)
        obs_data = None
        stored_h: Tensor | None = None
        turn_frac = 0.0

        # ── Learner step (collect trajectory) ───────────────────────────
        if isinstance(agent, StrategistAgent) and pid == work.learner_id:
            turn_frac = min(1.0, turn_number / max(1, work.max_turns))

            obs_data = encode_board(env.board, pid)
            data = obs_data.clone().to(agent.device)
            node_keys: list[str] = data.node_keys  # type: ignore[assignment]
            N = data.x.size(0)

            h_in = agent._get_aligned_h(node_keys, N, data.x.device)
            stored_h = h_in.cpu().clone()

            acting_mask_t = torch.tensor(
                [env.board[k].owner == pid and env.board[k].units > 0
                 for k in node_keys],
                dtype=torch.bool,
                device=agent.device,
            )

            with torch.no_grad():
                ml, alpha, beta, val, h_new = agent.model(
                    data.x, data.edge_index, data.edge_attr, data.u,
                    acting_mask_t, h_tiles=h_in, turn_frac=turn_frac, batch=None,
                )

            agent._h_tiles = h_new
            agent._node_keys = node_keys
            agent._turn = turn_number + 1

            src = data.edge_index[0]
            valid_src = acting_mask_t[src]
            ml = ml.masked_fill(~valid_src, -1e9)

            src_to_best: dict[int, int] = {}
            for ei in range(src.size(0)):
                s = src[ei].item()
                if not valid_src[ei]:
                    continue
                if s not in src_to_best or ml[ei] > ml[src_to_best[s]]:
                    src_to_best[s] = ei

            sel_edges = torch.tensor(list(src_to_best.values()), dtype=torch.long)
            if sel_edges.numel() > 0:
                frac_dist = _Beta(alpha[sel_edges], beta[sel_edges])
                sel_fracs = frac_dist.sample()
                lp_edge = F.log_softmax(ml, dim=0)[sel_edges]
                lp_frac = frac_dist.log_prob(sel_fracs.clamp(1e-6, 1 - 1e-6))
                # Keep per-order log probs [N_orders] — do NOT sum here.
                # The trainer clips each order's probability ratio independently.
                # Summing before storage produces a joint ratio that bypasses
                # PPO clipping (joint ratio = product of per-order ratios, which
                # can be orders of magnitude outside [1-ε, 1+ε]).
                log_p = lp_edge + lp_frac  # [N_orders]
            else:
                sel_fracs = torch.zeros(0)
                log_p = torch.zeros(0)  # [0] — empty, consistent with N_orders shape

            value_est = val
            chosen_edges = sel_edges
            chosen_fracs = sel_fracs

            dst = data.edge_index[1]
            for i, ei in enumerate(sel_edges.tolist()):
                from_key = node_keys[src[ei].item()]
                to_key = node_keys[dst[ei].item()]
                ft = env.board.get(from_key)
                if ft is None or ft.owner != pid or ft.units == 0:
                    continue
                units = max(1, round(sel_fracs[i].item() * ft.units))
                orders[from_key] = MovementOrder(
                    from_key=from_key, to_key=to_key, requested_units=units,
                )

        # ── Opponent step ────────────────────────────────────────────────
        else:
            orders = agent(env.board, pid, env.players, env.stats)

        result = resolve_player_turn(
            board=env.board,
            players=env.players,
            orders=orders,
            player_id=pid,
            stats=env.stats,
        )

        env.board = result.board
        env.players = result.players
        env.stats = result.stats

        for p in env.players:
            if p.is_eliminated and p.id not in env._elimination_order:
                env._elimination_order.append(p.id)

        reward = compute_step_reward(
            player_id=pid,
            events=result.events,
            board_before=board_before,
            board_after=env.board,
            players_before=players_before,
            players_after=env.players,
            winner_id=result.winner_id,
            config=work.reward_config,
        )

        done = result.winner_id is not None
        # Treat learner elimination as a terminal step even when the game
        # continues (no winner yet). Without this, the last learner transition
        # in an episode stores done=False, causing GAE to bleed across episode
        # boundaries: compute_returns() uses next_value from the NEXT episode's
        # first transition, corrupting advantage estimates for eliminated games.
        if pid == work.learner_id and not done:
            learner_after = next((p for p in env.players if p.id == pid), None)
            if learner_after is not None and learner_after.is_eliminated:
                done = True

        if pid == work.learner_id and obs_data is not None:
            tr = StrategistTransition(
                obs=obs_data,
                acting_mask=acting_mask_t.cpu(),
                chosen_edges=chosen_edges.cpu(),
                chosen_fractions=chosen_fracs.cpu(),
                log_prob=log_p.detach().cpu(),
                value=value_est.detach().cpu().squeeze(),
                reward=reward,
                done=done,
                h_tiles=stored_h,
                turn_frac=turn_frac,
            )
            transitions.append(_transition_to_raw(tr))

        if done:
            env._end_game(winner_id=result.winner_id)
            break

    return transitions


# ---------------------------------------------------------------------------
# Serial collector
# ---------------------------------------------------------------------------

class StrategistCollector:
    """
    Collect rollouts from league games for StrategistAgent training.

    Args:
        agent:          The StrategistAgent being trained.
        league:         League instance for opponent sampling.
        n_episodes:     Games per collect() call.
        max_turns:      Hard turn cap.
        reward_config:  Reward hyperparameters.
        seed:           Base RNG seed.
    """

    def __init__(
        self,
        agent: StrategistAgent,
        league: League,
        n_episodes: int = 16,
        max_turns: int = 300,
        reward_config: RewardConfig = DEFAULT_REWARD_CONFIG,
        seed: int = 0,
    ) -> None:
        self.agent = agent
        self.league = league
        self.n_episodes = n_episodes
        self.max_turns = max_turns
        self.reward_config = reward_config
        self.seed = seed
        self._call_count = 0   # advances base seed each collect() call

    def collect(self) -> StrategistRolloutBuffer:
        import random as _random
        # Advance base seed each call so every iteration sees different board
        # seeds and opponent assignments.  Using a fixed seed would cause the
        # same set of games to be collected on every iteration, potentially
        # overfitting to a tiny slice of the game distribution.
        base_seed = self.seed + self._call_count * self.n_episodes
        self._call_count += 1
        rng = _random.Random(base_seed)

        buf = StrategistRolloutBuffer()
        for ep in range(self.n_episodes):
            learner_id = rng.choice(PLAYER_IDS)
            work = self._make_work(ep, learner_id, base_seed)
            for raw in _run_episode_fn(work):
                buf.add(_raw_to_transition(raw))

        buf.compute_returns()
        return buf

    def _make_work(self, ep: int, learner_id: str, base_seed: int | None = None) -> _EpisodeWork:
        seed = (base_seed if base_seed is not None else self.seed) + ep
        sampled = self.league.sample_opponents(learner_id, PLAYER_IDS)
        learner_spec = _agent_to_spec(self.agent, self.max_turns)
        opponent_specs = {
            pid: (learner_spec if pid == learner_id else _agent_to_spec(ag, self.max_turns))
            for pid, ag in sampled.items()
        }
        return _EpisodeWork(
            seed=seed,
            learner_id=learner_id,
            learner_spec=learner_spec,
            opponent_specs=opponent_specs,
            max_turns=self.max_turns,
            reward_config=self.reward_config,
        )


# ---------------------------------------------------------------------------
# Parallel collector
# ---------------------------------------------------------------------------

class ParallelStrategistCollector:
    """
    Collect rollouts from league games using multiple CPU processes.

    Identical interface to StrategistCollector but runs episodes in parallel
    using ProcessPoolExecutor. Each worker calls torch.set_num_threads(1) to
    prevent thread-count explosion.

    Args:
        agent:          The StrategistAgent being trained.
        league:         League instance for opponent sampling.
        n_episodes:     Games per collect() call.
        max_turns:      Hard turn cap.
        n_workers:      Number of parallel processes.
                        Defaults to min(n_episodes, os.cpu_count()).
        reward_config:  Reward hyperparameters.
        seed:           Base RNG seed.
    """

    def __init__(
        self,
        agent: StrategistAgent,
        league: League,
        n_episodes: int = 16,
        max_turns: int = 300,
        n_workers: int | None = None,
        reward_config: RewardConfig = DEFAULT_REWARD_CONFIG,
        seed: int = 0,
    ) -> None:
        self.agent = agent
        self.league = league
        self.n_episodes = n_episodes
        self.max_turns = max_turns
        self.n_workers = n_workers or min(n_episodes, os.cpu_count() or 1)
        self.reward_config = reward_config
        self.seed = seed
        self._call_count = 0   # advances base seed each collect() call

    def collect(self) -> StrategistRolloutBuffer:
        """
        Run n_episodes games in parallel and return a filled buffer.

        No torch tensors cross process boundaries — work items carry numpy
        state-dicts and workers return plain dicts of numpy arrays.  Standard
        pickle handles everything; torch's shared-memory mechanism (file_system
        or file_descriptor) is never invoked, avoiding vm.max_map_count limits.

        We always use the 'spawn' multiprocessing context, even on Linux where
        'fork' is the default. Forking after CUDA has been initialised in the
        main process is unsafe (PyTorch explicitly documents this): workers
        inherit the CUDA context and can corrupt it on exit. 'spawn' starts
        each worker from a clean slate, adding ~100–200 ms startup overhead
        per collect() call — negligible vs the seconds spent on game simulation.
        """
        import multiprocessing
        import random as _random
        from concurrent.futures import ProcessPoolExecutor

        # Advance base seed each call so every iteration sees different board
        # seeds and opponent assignments.
        base_seed = self.seed + self._call_count * self.n_episodes
        self._call_count += 1
        rng = _random.Random(base_seed)

        # Pre-build all episode work packages in the main process
        works = []
        for ep in range(self.n_episodes):
            learner_id = rng.choice(PLAYER_IDS)
            works.append(self._make_work(ep, learner_id, base_seed))

        # Run in parallel subprocesses — force spawn to avoid CUDA fork hazard
        mp_context = multiprocessing.get_context("spawn")
        buf = StrategistRolloutBuffer()
        with ProcessPoolExecutor(max_workers=self.n_workers, mp_context=mp_context) as pool:
            for raw_list in pool.map(_run_episode_fn, works):
                for raw in raw_list:
                    buf.add(_raw_to_transition(raw))

        buf.compute_returns()
        return buf

    def _make_work(self, ep: int, learner_id: str, base_seed: int | None = None) -> _EpisodeWork:
        seed = (base_seed if base_seed is not None else self.seed) + ep
        sampled = self.league.sample_opponents(learner_id, PLAYER_IDS)
        learner_spec = _agent_to_spec(self.agent, self.max_turns)
        opponent_specs = {
            pid: (learner_spec if pid == learner_id else _agent_to_spec(ag, self.max_turns))
            for pid, ag in sampled.items()
        }
        return _EpisodeWork(
            seed=seed,
            learner_id=learner_id,
            learner_spec=learner_spec,
            opponent_specs=opponent_specs,
            max_turns=self.max_turns,
            reward_config=self.reward_config,
        )
