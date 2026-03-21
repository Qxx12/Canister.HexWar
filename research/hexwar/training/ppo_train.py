"""
PPO training entry point for HexWar.

Two-phase curriculum:
  Phase A — Bootstrap: learner plays against GreedyAgent opponents.
             Runs until win_rate > PHASE_A_THRESHOLD or PHASE_A_ITERS iters.
  Phase B — Self-play: opponents drawn from a growing pool of past checkpoints.
             Every SNAPSHOT_EVERY iterations the current policy is snapshotted
             into the opponent pool.

Usage::

    # Phase A then Phase B (default)
    python -m hexwar.training.ppo_train

    # Skip Phase A, go straight to self-play
    python -m hexwar.training.ppo_train --skip-phase-a

    # Resume from a checkpoint
    python -m hexwar.training.ppo_train --resume runs/ppo/ckpt_iter0050.pt
"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import torch

from ..agents.greedy_agent import GreedyAgent
from ..agents.neural.gnn_model import HexWarGNN
from ..agents.neural.ppo_agent import PPOAgent
from .ppo_trainer import PPOTrainer
from .self_play import SelfPlayCollector

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

HIDDEN_DIM       = 128
N_LAYERS         = 4
N_HEADS          = 4
HISTORY_K        = 5

LR               = 3e-4
CLIP_EPS         = 0.2
VF_COEF          = 0.5
ENT_COEF         = 0.01
MAX_GRAD_NORM    = 0.5
N_EPOCHS         = 4
BATCH_SIZE       = 64

N_EPISODES       = 16    # games per collect() call
MAX_TURNS        = 300

PHASE_A_ITERS    = 50    # max Phase A iterations before moving on
PHASE_A_THRESHOLD = 0.25  # win rate that triggers early Phase A exit
PHASE_B_ITERS    = 500   # Phase B iterations
SNAPSHOT_EVERY   = 20    # add learner to opponent pool every N iters
CHECKPOINT_EVERY = 10    # save checkpoint every N iters


def train(
    output_dir: str | Path = "runs/ppo",
    device: str = "cpu",
    n_episodes: int = N_EPISODES,
    max_turns: int = MAX_TURNS,
    phase_a_iters: int = PHASE_A_ITERS,
    phase_b_iters: int = PHASE_B_ITERS,
    skip_phase_a: bool = False,
    resume_from: str | Path | None = None,
    tensorboard: bool = True,
) -> None:
    """
    Run the full PPO training curriculum.

    Args:
        output_dir:     Directory for checkpoints and logs.
        device:         Torch device string ("cpu", "cuda", "mps").
        n_episodes:     Games per collect() call.
        max_turns:      Hard turn cap per game.
        phase_a_iters:  Max iterations for Phase A (bootstrap).
        phase_b_iters:  Iterations for Phase B (self-play).
        skip_phase_a:   Skip Phase A and go straight to self-play.
        resume_from:    Path to a .pt checkpoint to resume from.
        tensorboard:    Enable TensorBoard logging.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device)
    print(f"Device: {dev}", flush=True)

    # Build model + agent
    model = HexWarGNN(
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        history_k=HISTORY_K,
    )
    agent = PPOAgent(model=model, history_k=HISTORY_K, device=dev)

    trainer = PPOTrainer(
        agent=agent,
        lr=LR,
        clip_eps=CLIP_EPS,
        vf_coef=VF_COEF,
        ent_coef=ENT_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        device=dev,
    )

    start_iter = 0
    if resume_from:
        trainer.load_checkpoint(resume_from)
        meta_path = Path(resume_from).with_suffix(".json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            start_iter = meta.get("iteration", 0)
        print(f"Resumed from {resume_from} (iter {start_iter})", flush=True)

    if tensorboard:
        trainer.enable_tensorboard(str(out / "tb"))

    # ── Phase A: Bootstrap vs GreedyAgent ─────────────────────────────
    if not skip_phase_a:
        print("\n── Phase A: Bootstrap vs GreedyAgent ──", flush=True)
        greedy_opponents = [GreedyAgent() for _ in range(4)]
        collector_a = SelfPlayCollector(
            agent=agent,
            n_episodes=n_episodes,
            max_turns=max_turns,
            opponent_pool=greedy_opponents,
            seed=0,
        )

        for it in range(start_iter, phase_a_iters):
            t0 = time.perf_counter()
            collector_a.seed = it * n_episodes

            buffers = collector_a.collect()
            metrics = trainer.update(buffers)

            elapsed = time.perf_counter() - t0
            win_rate = _estimate_win_rate(buffers)
            print(
                f"[Phase A] iter {it + 1:4d}/{phase_a_iters} | "
                f"win%={win_rate * 100:.1f}  "
                f"loss={metrics.get('loss', 0):.4f}  "
                f"pg={metrics.get('pg_loss', 0):.4f}  "
                f"vf={metrics.get('vf_loss', 0):.4f}  "
                f"ent={metrics.get('entropy', 0):.4f}  "
                f"{elapsed:.0f}s",
                flush=True,
            )

            if (it + 1) % CHECKPOINT_EVERY == 0:
                _save(trainer, out, it + 1)

            if win_rate >= PHASE_A_THRESHOLD:
                print(f"  Phase A threshold reached at iter {it + 1}. Moving to Phase B.", flush=True)
                _save(trainer, out, it + 1, tag="phase_a_final")
                break

        start_iter = 0  # Phase B always counts from 0

    # ── Phase B: Self-play with opponent pool ─────────────────────────
    print("\n── Phase B: Self-play ──", flush=True)

    # Seed pool with a snapshot of the current policy
    initial_snapshot = _snapshot(agent, dev)
    collector_b = SelfPlayCollector(
        agent=agent,
        n_episodes=n_episodes,
        max_turns=max_turns,
        opponent_pool=[initial_snapshot],
        seed=10_000,
    )

    for it in range(start_iter, phase_b_iters):
        t0 = time.perf_counter()
        collector_b.seed = 10_000 + it * n_episodes

        buffers = collector_b.collect()
        metrics = trainer.update(buffers)

        elapsed = time.perf_counter() - t0
        win_rate = _estimate_win_rate(buffers)
        print(
            f"[Phase B] iter {it + 1:4d}/{phase_b_iters} | "
            f"win%={win_rate * 100:.1f}  "
            f"loss={metrics.get('loss', 0):.4f}  "
            f"pg={metrics.get('pg_loss', 0):.4f}  "
            f"vf={metrics.get('vf_loss', 0):.4f}  "
            f"ent={metrics.get('entropy', 0):.4f}  "
            f"{elapsed:.0f}s",
            flush=True,
        )

        # Snapshot current policy into opponent pool periodically
        if (it + 1) % SNAPSHOT_EVERY == 0:
            snap = _snapshot(agent, dev)
            collector_b.add_to_pool(snap)
            print(f"  Pool size: {len(collector_b.opponent_pool)}", flush=True)

        if (it + 1) % CHECKPOINT_EVERY == 0:
            _save(trainer, out, it + 1)

    _save(trainer, out, phase_b_iters, tag="final")
    print("\nTraining complete.", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snapshot(agent: PPOAgent, device: torch.device) -> PPOAgent:
    """Return a frozen copy of the agent for use as an opponent."""
    snap_model = HexWarGNN(
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        history_k=HISTORY_K,
    )
    snap_model.load_state_dict(deepcopy(agent.model.state_dict()))
    snap_model.eval()
    snap = PPOAgent(model=snap_model, history_k=HISTORY_K, deterministic=True, device=device)
    return snap


def _estimate_win_rate(buffers: dict) -> float:
    """Estimate win rate from the terminal rewards in the learner buffer."""
    wins = total = 0
    for buf in buffers.values():
        for tr in buf._transitions:
            if tr.done:
                total += 1
                if tr.reward > 0:
                    wins += 1
    return wins / total if total > 0 else 0.0


def _save(trainer: PPOTrainer, out: Path, iteration: int, tag: str = "") -> None:
    suffix = f"_{tag}" if tag else ""
    ckpt_path = out / f"ckpt_iter{iteration:04d}{suffix}.pt"
    trainer.save_checkpoint(ckpt_path)
    meta = {"iteration": iteration}
    with open(ckpt_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f)
    print(f"  ✓ checkpoint: {ckpt_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HexWar PPO agent")
    parser.add_argument("--output",       default="runs/ppo")
    parser.add_argument("--device",       default="cpu")
    parser.add_argument("--episodes",     type=int, default=N_EPISODES)
    parser.add_argument("--max-turns",    type=int, default=MAX_TURNS)
    parser.add_argument("--phase-a-iters", type=int, default=PHASE_A_ITERS)
    parser.add_argument("--phase-b-iters", type=int, default=PHASE_B_ITERS)
    parser.add_argument("--skip-phase-a", action="store_true")
    parser.add_argument("--resume",       default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()

    train(
        output_dir=args.output,
        device=args.device,
        n_episodes=args.episodes,
        max_turns=args.max_turns,
        phase_a_iters=args.phase_a_iters,
        phase_b_iters=args.phase_b_iters,
        skip_phase_a=args.skip_phase_a,
        resume_from=args.resume,
        tensorboard=not args.no_tensorboard,
    )
