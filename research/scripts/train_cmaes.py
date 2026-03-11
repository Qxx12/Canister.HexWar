#!/usr/bin/env python3
"""
Train GreedyAgent weights with CMA-ES evolutionary optimisation.

Usage:
    python scripts/train_cmaes.py --games 20 --generations 200

The output directory (default: runs/cmaes) will contain:
  - ckpt_gen####.json  — periodic checkpoints
  - best_weights.json  — final best weights

To resume from a checkpoint:
    python scripts/train_cmaes.py --resume runs/cmaes/ckpt_gen0100.json
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexwar.agents.evolutionary.cmaes_train import train

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CMA-ES GreedyAgent training")
    parser.add_argument("--output", default="runs/cmaes")
    parser.add_argument("--sigma0", type=float, default=0.5)
    parser.add_argument("--popsize", type=int, default=16)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    best = train(
        output_dir=args.output,
        sigma0=args.sigma0,
        popsize=args.popsize,
        max_generations=args.generations,
        n_games=args.games,
        resume_from=args.resume,
    )
    print("\nBest weights found:")
    for i, w in enumerate(best):
        print(f"  weight[{i}] = {w:.4f}")
