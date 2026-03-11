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
# Defaults — tuned for ~48h on M1 Pro (10 cores)
# ---------------------------------------------------------------------------

DEFAULT_SIGMA0    = 0.5
DEFAULT_POPSIZE   = 32    # λ: larger = more thorough search
DEFAULT_N_GAMES   = 60    # games per candidate per generation
DEFAULT_MAX_TURNS = 200
DEFAULT_N_WORKERS = 8     # leave 2 cores for OS / main process
MAX_GENERATIONS   = 1000  # effectively unlimited; stop via convergence


def train(
    output_dir: str | Path = "runs/cmaes",
    sigma0: float = DEFAULT_SIGMA0,
    popsize: int = DEFAULT_POPSIZE,
    max_generations: int = MAX_GENERATIONS,
    n_games: int = DEFAULT_N_GAMES,
    max_turns: int = DEFAULT_MAX_TURNS,
    n_workers: int = DEFAULT_N_WORKERS,
    resume_from: str | Path | None = None,
) -> list[float]:
    """
    Run CMA-ES optimisation and return the best weight vector found.

    Args:
        output_dir:      Directory for checkpoints and logs.
        sigma0:          Initial step size for CMA-ES.
        popsize:         Population size (λ).
        max_generations: Stop after this many generations.
        n_games:         Games per candidate per generation.
        max_turns:       Hard turn cap per game.
        n_workers:       Parallel worker processes.
        resume_from:     Path to a JSON checkpoint to continue from.

    Returns:
        Best weight vector as a list of floats.
    """
    if not CMA_AVAILABLE:
        raise ImportError("cma not installed. Run: pip install cma")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    x0 = list(DEFAULT_WEIGHTS)
    start_gen = 0
    if resume_from:
        with open(resume_from) as f:
            ckpt = json.load(f)
        x0 = ckpt["best_weights"]
        start_gen = ckpt.get("generation", 0)
        print(f"Resuming from {resume_from} (gen {start_gen}, score={ckpt['best_score']:.4f})", flush=True)

    opts = cma.CMAOptions()
    opts["popsize"]  = popsize
    opts["maxiter"]  = max_generations
    opts["verbose"]  = -9   # silence CMA's own output; we log ourselves

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    best_weights = x0[:]
    best_score   = -float("inf")
    gen          = start_gen

    games_per_gen = popsize * n_games
    print(f"CMA-ES  dim={N_FEATURES}  popsize={popsize}  sigma0={sigma0}", flush=True)
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
    parser.add_argument("--output",      default="runs/cmaes")
    parser.add_argument("--sigma0",      type=float, default=DEFAULT_SIGMA0)
    parser.add_argument("--popsize",     type=int,   default=DEFAULT_POPSIZE)
    parser.add_argument("--generations", type=int,   default=MAX_GENERATIONS)
    parser.add_argument("--games",       type=int,   default=DEFAULT_N_GAMES)
    parser.add_argument("--max-turns",   type=int,   default=DEFAULT_MAX_TURNS)
    parser.add_argument("--workers",     type=int,   default=DEFAULT_N_WORKERS)
    parser.add_argument("--resume",      default=None)
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
    )
