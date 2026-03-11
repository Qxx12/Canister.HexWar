#!/usr/bin/env python3
"""
Train PPO neural agent via self-play.

Usage:
    python scripts/train_ppo.py --iterations 1000 --episodes 16

Requires: torch, torch_geometric

Output directory (default: runs/ppo) will contain:
  - ckpt_iter####.pt  — periodic checkpoints
  - tensorboard/      — TensorBoard logs (run: tensorboard --logdir runs/ppo)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="PPO HexWar agent training")
    parser.add_argument("--output", default="runs/ppo")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=16, help="Episodes per collection phase")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=50)
    args = parser.parse_args()

    try:
        from hexwar.agents.neural.gnn_model import HexWarGNN
        from hexwar.agents.neural.ppo_agent import PPOAgent
        from hexwar.training.ppo_trainer import PPOTrainer
        from hexwar.training.self_play import SelfPlayCollector
    except ImportError as e:
        print(f"Import error: {e}")
        print("Install dependencies: pip install torch torch_geometric")
        sys.exit(1)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = HexWarGNN(hidden_dim=args.hidden_dim, n_layers=args.n_layers)
    agent = PPOAgent(model=model, device=device)

    trainer = PPOTrainer(
        agent=agent,
        lr=args.lr,
        clip_eps=args.clip_eps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=device,
    )
    trainer.enable_tensorboard(str(out / "tensorboard"))

    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from {args.resume}")

    collector = SelfPlayCollector(
        agent=agent,
        n_episodes=args.episodes,
        max_turns=args.max_turns,
    )

    print(f"Starting PPO training: {args.iterations} iterations, {args.episodes} episodes each")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    for iteration in range(1, args.iterations + 1):
        # Collect rollouts
        collector.seed = iteration * 1000
        buffers = collector.collect()

        # Update policy
        metrics = trainer.update(buffers)

        if iteration % args.log_interval == 0:
            n_transitions = sum(len(b) for b in buffers.values())
            print(
                f"Iter {iteration:5d} | "
                f"transitions={n_transitions:5d} | "
                f"loss={metrics.get('loss', 0):.4f} | "
                f"pg={metrics.get('pg_loss', 0):.4f} | "
                f"ent={metrics.get('entropy', 0):.4f}"
            )

        if iteration % args.save_interval == 0:
            ckpt_path = out / f"ckpt_iter{iteration:05d}.pt"
            trainer.save_checkpoint(ckpt_path)
            print(f"  Saved: {ckpt_path}")

    # Final checkpoint
    trainer.save_checkpoint(out / "final.pt")
    print(f"\nTraining complete. Final checkpoint: {out}/final.pt")


if __name__ == "__main__":
    main()
