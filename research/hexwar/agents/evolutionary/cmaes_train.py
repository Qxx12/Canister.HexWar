"""
CMA-ES training for GreedyAgent weights.

Uses the `cma` library (Hansen 2016) to optimise the 8-dimensional weight
vector of GreedyAgent via the (μ, λ)-CMA-ES algorithm.

Parallelism: each generation evaluates all popsize×n_games games in parallel
across CPU cores via evaluate_generation(). On M1 Pro (10 cores, 8 workers)
a generation with popsize=32, n_games=60 takes ~10 min, giving ~260 gens in 48h.

Usage::

    python -m hexwar.agents.evolutionary.cmaes_train

Or import and call train() directly for custom configurations.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

try:
    import cma
    CMA_AVAILABLE = True
except ImportError:
    CMA_AVAILABLE = False

from ..greedy_agent import DEFAULT_WEIGHTS, N_FEATURES
from .fitness import FitnessResult, evaluate_generation

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SIGMA0    = 0.5
DEFAULT_POPSIZE   = 32    # λ: larger = more thorough search
DEFAULT_N_GAMES   = 100   # increased from 60 — halves fitness noise
DEFAULT_MAX_TURNS = 200
DEFAULT_N_WORKERS = 8     # leave 2 cores for OS / main process
MAX_GENERATIONS   = 1000  # effectively unlimited; stop via convergence

# CMA-ES search dimension: N_FEATURES weight dims + 1 send_fraction dim
SEARCH_DIM = N_FEATURES + 1   # 15

# Initial send_fraction (last element of x0).
# CMA-ES will optimise it; clamped to [0.5, 1.0] during evaluation.
SEND_FRACTION_INIT = 1.0

# Initial weights for features 8-13 that don't exist in older checkpoints.
# Set at a scale meaningful relative to the evolved v3 weights (-9 to +12).
#   8  inv_dist_to_unowned_start: strong path pull toward win-condition tiles
#   9  target_owner_near_elim:    elimination bonus, should dominate neutral
#  10  neutral_adj_to_target:     junction tile bonus
#  11  source_threat_ratio:       negative — discourages attacking from threatened positions
#  12  is_gateway:                binary last-hop bonus (on top of gradient feature 8)
#  13  enemy_adj_to_own_start:    very high — clearing threats to our start is critical
NEW_FEATURE_INIT: list[float] = [7.0, 5.0, 4.0, -5.0, 5.0, 10.0]


def train(
    output_dir: str | Path = "runs/cmaes",
    sigma0: float = DEFAULT_SIGMA0,
    popsize: int = DEFAULT_POPSIZE,
    max_generations: int = MAX_GENERATIONS,
    n_games: int = DEFAULT_N_GAMES,
    max_turns: int = DEFAULT_MAX_TURNS,
    n_workers: int = DEFAULT_N_WORKERS,
    resume_from: str | Path | None = None,
    warmstart_from: str | Path | None = None,
    opponent_ckpt: str | Path | None = None,
    opponent_mix_ratio: float = 0.6,
) -> list[float]:
    """
    Run CMA-ES optimisation and return the best weight vector found.

    Args:
        output_dir:          Directory for checkpoints and logs.
        sigma0:              Initial step size for CMA-ES.
        popsize:             Population size (λ).
        max_generations:     Stop after this many generations.
        n_games:             Games per candidate per generation.
        max_turns:           Hard turn cap per game.
        n_workers:           Parallel worker processes.
        resume_from:         Path to a v4 JSON checkpoint to *continue* a run
                             (restores x0 from that checkpoint; SEARCH_DIM must
                             match).
        warmstart_from:      Path to an *older* checkpoint (e.g. v3 8-dim) to
                             use as the starting point for the 8 original dims.
                             The 3 new feature dims are initialised to
                             NEW_FEATURE_INIT and send_fraction to 1.0.
                             Use this to start v4 from the v3 best result.
        opponent_ckpt:       Path to a checkpoint whose best_weights are used
                             as the evolved-opponent pool. Opponents are a mix
                             of this checkpoint and DEFAULT_WEIGHTS according
                             to opponent_mix_ratio.
        opponent_mix_ratio:  Fraction of the 5 opponents drawn from
                             opponent_ckpt (rounded). Default 0.6 → 3 evolved
                             + 2 default. 1.0 → full self-play against the
                             checkpoint.

    Returns:
        Best weight vector as a list of floats (SEARCH_DIM elements).
    """
    if not CMA_AVAILABLE:
        raise ImportError("cma not installed. Run: pip install cma")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ x0
    start_gen = 0
    if resume_from:
        # Continue an interrupted v4 run — expects SEARCH_DIM weights
        with open(resume_from) as f:
            ckpt = json.load(f)
        x0 = ckpt["best_weights"]
        start_gen = ckpt.get("generation", 0)
        print(
            f"Resuming from {resume_from} "
            f"(gen {start_gen}, score={ckpt['best_score']:.4f})",
            flush=True,
        )
    elif warmstart_from:
        # Start v4 from an older (possibly 8-dim) checkpoint.
        # Zero-pads missing feature dims, then appends NEW_FEATURE_INIT and
        # send_fraction so the new dims begin at sensible values.
        with open(warmstart_from) as f:
            ckpt = json.load(f)
        old_w = ckpt["best_weights"]
        # Pad to N_FEATURES if shorter (e.g. v3 had 8 features)
        if len(old_w) < N_FEATURES:
            old_w = old_w + NEW_FEATURE_INIT[: N_FEATURES - len(old_w)]
        x0 = old_w[:N_FEATURES] + [SEND_FRACTION_INIT]
        print(
            f"Warmstart from {warmstart_from} "
            f"(score={ckpt['best_score']:.4f}), new dims → {NEW_FEATURE_INIT}",
            flush=True,
        )
    else:
        x0 = list(DEFAULT_WEIGHTS) + [SEND_FRACTION_INIT]

    # ------------------------------------------------------------------ opponent pool
    opponent_weights: list[list[float]] | None = None
    if opponent_ckpt:
        with open(opponent_ckpt) as f:
            opp_data = json.load(f)
        evolved_w = opp_data["best_weights"]   # may be 8-dim or 11-dim
        n_evolved = max(1, round(5 * opponent_mix_ratio))
        n_default = 5 - n_evolved
        opponent_weights = (
            [list(evolved_w)] * n_evolved
            + [list(DEFAULT_WEIGHTS)] * n_default
        )
        print(
            f"Opponent pool: {n_evolved}× evolved (from {opponent_ckpt}) "
            f"+ {n_default}× DEFAULT_WEIGHTS",
            flush=True,
        )

    opts = cma.CMAOptions()
    opts["popsize"]  = popsize
    opts["maxiter"]  = max_generations
    opts["verbose"]  = -9   # silence CMA's own output; we log ourselves

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    best_weights = x0[:]
    best_score   = -float("inf")
    gen          = start_gen

    games_per_gen = popsize * n_games
    print(f"CMA-ES  dim={SEARCH_DIM}  popsize={popsize}  sigma0={sigma0}", flush=True)
    print(f"Games/gen={games_per_gen}  max_turns={max_turns}  workers={n_workers}", flush=True)
    print(f"Output: {out.resolve()}", flush=True)
    print("-" * 60, flush=True)

    run_start = time.perf_counter()

    while not es.stop():
        gen_start = time.perf_counter()

        solutions = es.ask()          # list of np arrays (λ candidates)
        seed_offset = gen * games_per_gen

        # Evaluate all candidates in parallel
        fitness_results: list[FitnessResult] = evaluate_generation(
            all_weights=[list(w) for w in solutions],
            n_games=n_games,
            max_turns=max_turns,
            seed_offset=seed_offset,
            n_workers=n_workers,
            opponent_weights=opponent_weights,
        )

        scores   = [r.score for r in fitness_results]
        fitnesses = [-s for s in scores]   # CMA-ES minimises

        es.tell(solutions, fitnesses)

        gen_time = time.perf_counter() - gen_start
        gen += 1

        # Track best
        for _i, (r, w) in enumerate(zip(fitness_results, solutions)):
            if r.score > best_score:
                best_score   = r.score
                best_weights = list(w)

        avg_score  = sum(scores) / len(scores)
        best_this  = max(scores)
        avg_wins   = sum(r.win_rate for r in fitness_results) / len(fitness_results)
        avg_tiles  = sum(r.avg_tiles_held for r in fitness_results) / len(fitness_results)
        elapsed_h  = (time.perf_counter() - run_start) / 3600

        print(
            f"Gen {gen:4d} | "
            f"best_all={best_score:.4f}  this={best_this:.4f}  avg={avg_score:.4f} | "
            f"win%={avg_wins*100:.1f}  tiles={avg_tiles:.1f} | "
            f"σ={es.sigma:.4f}  {gen_time:.0f}s  [{elapsed_h:.1f}h]",
            flush=True,
        )

        # Checkpoint every 10 generations
        if gen % 10 == 0:
            ckpt_path = out / f"ckpt_gen{gen:04d}.json"
            _save_checkpoint(ckpt_path, best_weights, best_score, gen)

        if es.stop():
            break

    _save_checkpoint(out / "best_weights.json", best_weights, best_score, gen)
    total_h = (time.perf_counter() - run_start) / 3600
    print(f"\nDone — {gen} generations in {total_h:.2f}h", flush=True)
    print(f"Best score: {best_score:.4f}", flush=True)
    print(f"Best weights: {[round(w, 4) for w in best_weights]}", flush=True)
    return best_weights


def _save_checkpoint(path: Path, weights: list[float], score: float, gen: int) -> None:
    with open(path, "w") as f:
        json.dump({"generation": gen, "best_score": score, "best_weights": weights}, f, indent=2)
    print(f"  ✓ checkpoint: {path}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train GreedyAgent weights with CMA-ES")
    parser.add_argument("--output",        default="runs/cmaes")
    parser.add_argument("--sigma0",        type=float, default=DEFAULT_SIGMA0)
    parser.add_argument("--popsize",       type=int,   default=DEFAULT_POPSIZE)
    parser.add_argument("--generations",   type=int,   default=MAX_GENERATIONS)
    parser.add_argument("--games",         type=int,   default=DEFAULT_N_GAMES)
    parser.add_argument("--max-turns",     type=int,   default=DEFAULT_MAX_TURNS)
    parser.add_argument("--workers",       type=int,   default=DEFAULT_N_WORKERS)
    parser.add_argument("--resume",        default=None,
                        help="Continue a v4 run from a SEARCH_DIM checkpoint")
    parser.add_argument("--warmstart",     default=None,
                        help="Start v4 from an older checkpoint (e.g. v3 8-dim); "
                             "new feature dims are initialised to NEW_FEATURE_INIT")
    parser.add_argument("--opponent-ckpt", default=None,
                        help="Checkpoint whose weights populate the evolved-opponent pool")
    parser.add_argument("--mix-ratio",     type=float, default=0.6,
                        help="Fraction of 5 opponents from --opponent-ckpt (default 0.6 → 3 evolved + 2 default)")
    args = parser.parse_args()

    train(
        output_dir=args.output,
        sigma0=args.sigma0,
        popsize=args.popsize,
        max_generations=args.generations,
        n_games=args.games,
        max_turns=args.max_turns,
        n_workers=args.workers,
        resume_from=args.resume,
        warmstart_from=args.warmstart,
        opponent_ckpt=args.opponent_ckpt,
        opponent_mix_ratio=args.mix_ratio,
    )
