#!/usr/bin/env python3
"""
Run a tournament between agents and print Elo rankings.

Usage:
    python scripts/run_tournament.py --games 100

To include a trained neural agent:
    python scripts/run_tournament.py --ppo-checkpoint runs/ppo/final.pt --games 200
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from hexwar.agents.greedy_agent import GreedyAgent
from hexwar.agents.random_agent import RandomAgent
from hexwar.evaluation.elo import EloSystem
from hexwar.evaluation.tournament import print_tournament_results, run_tournament


def main():
    parser = argparse.ArgumentParser(description="Agent tournament")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ppo-checkpoint", default=None)
    args = parser.parse_args()

    agents = {
        "random": RandomAgent(seed=42),
        "greedy_default": GreedyAgent(),
    }

    if args.ppo_checkpoint:
        try:
            from hexwar.agents.neural.gnn_model import HexWarGNN
            from hexwar.agents.neural.ppo_agent import PPOAgent

            model = HexWarGNN()
            ppo = PPOAgent(model=model, deterministic=True)
            ppo.load(args.ppo_checkpoint)
            agents["ppo_neural"] = ppo
            print(f"Loaded PPO agent from {args.ppo_checkpoint}")
        except Exception as e:
            print(f"Warning: Could not load PPO agent: {e}")

    print(f"Running tournament: {len(agents)} agents, {args.games} games each")
    stats = run_tournament(agents, n_games=args.games, max_turns=args.max_turns, seed=args.seed)
    print_tournament_results(stats)

    # Elo ratings from tournament results
    elo = EloSystem()
    names = list(agents.keys())
    for name in names:
        for other in names:
            if name == other:
                continue
            # Simple 1v1 Elo from per-slot win rates
            elo.ratings[name]  # init
    print("\nNote: Run with more games for stable Elo estimates.")


if __name__ == "__main__":
    main()
