"""
Fitness evaluation for evolutionary agent optimisation.

The fitness function runs N games with a candidate weight vector (as a
GreedyAgent) against a pool of opponents and returns a scalar score.

Design choices:
  - Opponents: 5 greedy agents using DEFAULT_WEIGHTS. All-greedy opposition
    is a much harder and more discriminating baseline than random opponents —
    only genuinely better weight vectors win consistently.
  - Slot rotation: the candidate cycles through all 6 player slots evenly
    across its N games (game_idx % 6), removing positional bias.
  - Score components (weighted sum):
      win_rate      — fraction of games won (primary signal)
      avg_turns     — lower is better when winning (tiebreak)
      avg_tiles     — tiles held at game end (proxy for board control)
  - Parallelism: a generation's full workload (popsize × n_games) is
    submitted as individual tasks to a ProcessPoolExecutor, so all CPU
    cores stay busy throughout the generation.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from ...engine.game_engine import HexWarEnv
from ...engine.types import PLAYER_IDS
from ..greedy_agent import DEFAULT_WEIGHTS, N_FEATURES, GreedyAgent


@dataclass
class FitnessResult:
    win_rate: float
    avg_turns: float
    avg_tiles_held: float
    score: float       # final scalar sent to CMA-ES


# ---------------------------------------------------------------------------
# Single-game task (must be a module-level function for pickling)
# ---------------------------------------------------------------------------

def _run_one_game(
    weights: list[float],
    seed: int,
    max_turns: int,
    win_weight: float,
    turns_weight: float,
    tiles_weight: float,
    candidate_slot: int = 0,
    opponent_weights: list[list[float]] | None = None,
) -> tuple[float, float, float]:
    """
    Run one game and return (win, turns, tiles_at_end).
    Module-level so it can be pickled by multiprocessing.

    weights: full search vector for the candidate (N_FEATURES + 1 dims if
             send_fraction is included; _build_agents splits it).
    opponent_weights: list of 5 weight vectors, one per non-candidate slot.
                      None → all opponents use DEFAULT_WEIGHTS.
    """
    candidate_id = PLAYER_IDS[candidate_slot]
    agents = _build_agents(weights, candidate_id, opponent_weights)
    env = HexWarEnv(agents=agents, seed=seed, max_turns=max_turns)
    result = env.run()

    won = float(result.winner_id == candidate_id)
    turns = float(result.turns_played)
    tiles = float(sum(1 for t in env.board.values() if t.owner == candidate_id))
    return won, turns, tiles


# ---------------------------------------------------------------------------
# Per-candidate evaluation (sequential, used when n_workers=1)
# ---------------------------------------------------------------------------

def evaluate_weights(
    weights: Sequence[float],
    n_games: int = 20,
    max_turns: int = 200,
    seed_offset: int = 0,
    win_weight: float = 1.0,
    turns_weight: float = 0.001,
    tiles_weight: float = 0.002,
) -> FitnessResult:
    """
    Evaluate a weight vector by running N_GAMES games sequentially.

    Used by the parallel evaluator to aggregate results. Can also be called
    directly for single-candidate evaluation.
    """
    wins = turns_total = tiles_total = 0.0
    for game_idx in range(n_games):
        w, t, tl = _run_one_game(
            list(weights), seed_offset + game_idx, max_turns,
            win_weight, turns_weight, tiles_weight,
            candidate_slot=game_idx % len(PLAYER_IDS),
        )
        wins += w
        turns_total += t
        tiles_total += tl

    win_rate = wins / n_games
    avg_turns = turns_total / n_games
    avg_tiles = tiles_total / n_games
    score = win_weight * win_rate - turns_weight * avg_turns + tiles_weight * avg_tiles

    return FitnessResult(
        win_rate=win_rate,
        avg_turns=avg_turns,
        avg_tiles_held=avg_tiles,
        score=score,
    )


# ---------------------------------------------------------------------------
# Parallel generation evaluation
# ---------------------------------------------------------------------------

def evaluate_generation(
    all_weights: list[list[float]],
    n_games: int = 60,
    max_turns: int = 200,
    seed_offset: int = 0,
    n_workers: int | None = None,
    win_weight: float = 1.0,
    turns_weight: float = 0.001,
    tiles_weight: float = 0.002,
    opponent_weights: list[list[float]] | None = None,
) -> list[FitnessResult]:
    """
    Evaluate an entire generation (all candidates) in parallel.

    Submits every (candidate, game) pair as an independent task so all
    CPU cores stay saturated throughout the generation. Results are
    aggregated per candidate after all tasks complete.

    Args:
        all_weights:      List of search vectors (one per candidate).
                          Each vector is N_FEATURES+1 dims (weights + send_fraction).
        n_games:          Games per candidate.
        max_turns:        Hard turn cap per game.
        seed_offset:      Base seed; candidate i game j uses seed_offset+i*n_games+j.
        n_workers:        Number of parallel worker processes. Defaults to
                          min(cpu_count, 8) — leaves 2 cores for the main process.
        win_weight, turns_weight, tiles_weight: Score formula coefficients.
        opponent_weights: 5-element list of weight vectors for the 5 opponent
                          agents. None → all opponents use DEFAULT_WEIGHTS.

    Returns:
        List of FitnessResult, one per candidate (same order as all_weights).
    """
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 8)

    popsize = len(all_weights)

    # Build the full task list: (candidate_idx, game_idx) → future
    futures: dict = {}
    # Accumulate results per candidate
    accum: list[list[tuple[float, float, float]]] = [[] for _ in range(popsize)]

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for cand_idx, weights in enumerate(all_weights):
            for game_idx in range(n_games):
                seed = seed_offset + cand_idx * n_games + game_idx
                fut = pool.submit(
                    _run_one_game,
                    list(weights), seed, max_turns,
                    win_weight, turns_weight, tiles_weight,
                    game_idx % len(PLAYER_IDS),
                    opponent_weights,
                )
                futures[fut] = cand_idx

        for fut in as_completed(futures):
            cand_idx = futures[fut]
            accum[cand_idx].append(fut.result())

    # Aggregate
    results = []
    for games in accum:
        n = len(games)
        win_rate  = sum(g[0] for g in games) / n
        avg_turns = sum(g[1] for g in games) / n
        avg_tiles = sum(g[2] for g in games) / n
        score = win_weight * win_rate - turns_weight * avg_turns + tiles_weight * avg_tiles
        results.append(FitnessResult(
            win_rate=win_rate,
            avg_turns=avg_turns,
            avg_tiles_held=avg_tiles,
            score=score,
        ))
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_agents(
    weights: Sequence[float],
    candidate_id: str,
    opponent_weights: list[list[float]] | None = None,
) -> dict:
    """
    Build the agent dict: candidate vs 5 opponents.

    weights: full CMA-ES search vector. First N_FEATURES elements are feature
             weights; element N_FEATURES (if present) is send_fraction, clamped
             to [0.5, 1.0].
    opponent_weights: 5 weight vectors (one per non-candidate slot). Each may
                      be shorter than N_FEATURES (e.g. an old 8-dim checkpoint)
                      — zero-padded to N_FEATURES automatically. None → all
                      opponents use DEFAULT_WEIGHTS at send_fraction=1.0.
    """
    feature_w = list(weights[:N_FEATURES])
    send_frac = float(max(0.5, min(1.0, weights[N_FEATURES]))) \
        if len(weights) > N_FEATURES else 1.0
    agents = {candidate_id: GreedyAgent(weights=feature_w, send_fraction=send_frac)}

    opp_idx = 0
    for pid in PLAYER_IDS:
        if pid == candidate_id:
            continue
        if opponent_weights is not None:
            raw = list(opponent_weights[opp_idx])
            # Zero-pad if checkpoint was trained with fewer features (e.g. v3 8-dim)
            if len(raw) < N_FEATURES:
                raw += [0.0] * (N_FEATURES - len(raw))
            agents[pid] = GreedyAgent(weights=raw[:N_FEATURES])
        else:
            agents[pid] = GreedyAgent(weights=list(DEFAULT_WEIGHTS))
        opp_idx += 1
    return agents
