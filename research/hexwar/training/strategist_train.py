"""
StrategistGNN training entry point.

Three-phase curriculum:
  Phase A — Bootstrap vs GreedyAgent:
             Builds basic strategic intuition before self-play begins.
             Runs until win_rate > PHASE_A_THRESHOLD or PHASE_A_ITERS iters.
  Phase B — League self-play:
             Opponents drawn from a growing pool of snapshots + evolved greedy
             agents.  Every SNAPSHOT_EVERY updates the current policy is added
             to the snapshot pool.
  Phase C — Competitive: same as B but with higher entropy penalty reduction
             to encourage more decisive play.

What makes StrategistGNN superior:
  - Per-tile GRU hidden state: remembers attack/defense patterns over all past
    turns (infinite temporal horizon vs. 5-frame stacking).
  - Global self-attention: every tile sees the full board before acting
    (eliminates the hop-distance blind spot of message-passing alone).
  - Phase input: the network knows early/mid/late game automatically.
  - League training: diverse opponents prevent exploitation of fixed strategies.

Usage::

    python -m hexwar.training.strategist_train
    python -m hexwar.training.strategist_train --resume runs/strategist/ckpt_iter0050.pt
    python -m hexwar.training.strategist_train --device cuda --n-episodes 32
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam

from ..agents.greedy_agent import GreedyAgent
from ..agents.neural.strategist_agent import StrategistAgent
from ..agents.neural.strategist_model import StrategistGNN
from .behavioural_cloning import bc_train
from .league import League, LeagueConfig
from .strategist_collect import (
    ParallelStrategistCollector,
    StrategistCollector,
    StrategistTransition,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

HIDDEN_DIM        = 128
N_LAYERS          = 4
N_HEADS           = 4
DROPOUT           = 0.1
MAX_TURNS         = 300

LR                = 3e-4
CLIP_EPS          = 0.2
VF_COEF           = 0.5
ENT_COEF          = 0.0
MAX_GRAD_NORM     = 0.5
N_EPOCHS          = 4
BATCH_SIZE        = 64
TARGET_KL         = 0.05   # approx KL threshold for early epoch stopping

N_EPISODES        = 16    # games per collect() call

BC_GAMES          = 200   # greedy games for behavioural cloning warm-start
BC_EPOCHS         = 10    # BC training epochs
PHASE_A_ITERS     = 50    # greedy bootstrap iterations
PHASE_A_THRESHOLD = 0.25  # win rate triggers early Phase A exit
PHASE_B_ITERS     = 300   # league self-play (normal entropy)
PHASE_C_ITERS     = 200   # competitive (reduced entropy — encourages decisiveness)
TOTAL_ITERS       = PHASE_B_ITERS + PHASE_C_ITERS   # 500
ENT_COEF_C        = 0.0    # Phase C: no entropy regularization (same as B)
SNAPSHOT_EVERY    = 25    # iterations between league snapshots
EVAL_EVERY        = 10    # iterations between evaluations
EVAL_GAMES        = 12


# ---------------------------------------------------------------------------
# PPO update for StrategistTransition
# ---------------------------------------------------------------------------

class StrategistTrainer:
    """
    PPO trainer adapted for StrategistAgent (GRU state re-injection).

    Identical to PPOTrainer logic but calls agent.evaluate_actions with
    the h_tiles and turn_frac stored in each StrategistTransition.
    """

    def __init__(
        self,
        agent: StrategistAgent,
        lr: float = LR,
        clip_eps: float = CLIP_EPS,
        vf_coef: float = VF_COEF,
        ent_coef: float = ENT_COEF,
        max_grad_norm: float = MAX_GRAD_NORM,
        n_epochs: int = N_EPOCHS,
        batch_size: int = BATCH_SIZE,
        target_kl: float = TARGET_KL,
        device: str | torch.device = "cpu",
    ) -> None:
        self.agent = agent
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.target_kl = target_kl
        self.device = torch.device(device)
        self.agent.model = self.agent.model.to(self.device)
        self.optimizer = Adam(agent.model.parameters(), lr=lr)
        self._step = 0
        self._writer = None

    def update(self, buf: StrategistRolloutBuffer) -> dict[str, float]:  # noqa: F821
        all_trans = list(buf._transitions)
        if not all_trans:
            return {}

        # Guard: if model weights are already corrupted (NaN/Inf), skip the
        # entire update rather than making things worse.
        if any(not p.data.isfinite().all() for p in self.agent.model.parameters()):
            print("WARNING: model weights contain NaN/Inf — skipping update", flush=True)
            return {}

        total = {"loss": 0.0, "pg": 0.0, "vf": 0.0, "ent": 0.0}
        n = 0

        # Do NOT switch to train() mode.  Dropout must be disabled here to keep
        # evaluate_actions() consistent with collection-time (eval-mode) log probs.
        # With dropout active, the same weights produce different logits each call,
        # making ratio = exp(new_log_prob - old_log_prob) arbitrarily large.
        # When adv < 0 and ratio >> 1, PPO clipping does NOT bound the loss:
        #   min(ratio*adv, clip(ratio)*adv) selects the unclipped (more negative)
        #   value, giving loss = -ratio*adv → catastrophic divergence.
        # model.eval() only disables dropout/batchnorm — autograd and optimizer
        # still work correctly in eval mode.

        for epoch in range(self.n_epochs):
            import random
            random.shuffle(all_trans)
            epoch_kl = 0.0
            epoch_batches = 0
            for start in range(0, len(all_trans), self.batch_size):
                batch = all_trans[start : start + self.batch_size]
                m = self._update_batch(batch)
                for k in total:
                    total[k] += m.get(k, 0.0)
                n += 1
                epoch_kl += m.get("approx_kl", 0.0)
                epoch_batches += 1

            # KL early stopping: if the mean approximate KL divergence across
            # all batches in this epoch exceeds target_kl, the policy has drifted
            # far enough from the collection-time policy — stop re-using these
            # transitions.  This prevents the feedback loop where concentrated
            # Beta distributions create large log_ratios → huge gradients →
            # even more concentrated distributions → runaway loss explosion.
            if epoch_batches > 0 and epoch_kl / epoch_batches > self.target_kl:
                break

        self.agent.model.eval()
        self._step += 1

        averaged = {k: v / max(n, 1) for k, v in total.items()}
        result = {
            "loss":    averaged["loss"],
            "pg_loss": averaged["pg"],
            "vf_loss": averaged["vf"],
            "entropy": averaged["ent"],
        }

        if self._writer:
            for k, v in result.items():
                self._writer.add_scalar(f"train/{k}", v, self._step)

        return result

    def _update_batch(self, batch: list[StrategistTransition]) -> dict[str, float]:
        raw_adv = torch.stack([tr.advantage for tr in batch])
        # Use correction=0 (population std) so a batch of size 1 returns 0.0
        # instead of NaN.  With correction=1 (default Bessel), std([x]) is
        # sqrt(0/0) = NaN, which then propagates: norm_adv → adv → loss →
        # loss.backward() → NaN in every parameter → model permanently corrupted.
        norm_adv = (raw_adv - raw_adv.mean()) / (raw_adv.std(correction=0) + 1e-8)

        pg_losses, vf_losses, entropies, log_ratios = [], [], [], []

        for i, tr in enumerate(batch):
            if tr.chosen_edges.numel() == 0:
                continue

            obs = tr.obs.to(self.device)
            am = tr.acting_mask.to(self.device)
            ce = tr.chosen_edges.to(self.device)
            cf = tr.chosen_fractions.to(self.device)
            old_lp = tr.log_prob.to(self.device)
            ret = tr.return_.to(self.device)
            adv = norm_adv[i].to(self.device)
            h = tr.h_tiles.to(self.device) if tr.h_tiles is not None else None

            try:
                log_prob, entropy, value = self.agent.evaluate_actions(
                    obs, am, ce, cf, h_tiles=h, turn_frac=tr.turn_frac
                )
            except Exception:
                # Beta(NaN, NaN) raises ValueError if weights are corrupted.
                # Skip this transition rather than crashing the whole update.
                continue

            # Per-order PPO clipping.
            # log_prob and old_lp are both [N_orders] — one entry per movement
            # order issued this turn. Clipping each order independently bounds
            # the per-order ratio to [1-ε, 1+ε], preventing the joint ratio
            # (which is the product of all per-order ratios) from exploding.
            # The previous joint approach (exp(log_prob.sum() - old_lp.sum()))
            # produced ratios like 1.2^10 ≈ 6x for a 10-order turn, which PPO
            # clipping cannot contain and leads to catastrophic loss spikes.
            #
            # Clamp log-ratio to [-1, 1] before exp.  exp(1) ≈ 2.7x max ratio.
            # Tighter than [-3, 3] because when adv < 0 and ratio > 1+ε,
            # PPO clipping does NOT bound the loss:
            #   min(ratio*adv, clip(ratio)*adv) selects ratio*adv (unclipped).
            # With exp(3)≈20x and moderate advantages, pg_loss of 2-5 builds up
            # slowly over 200 iters, causing policy drift and win-rate collapse.
            # exp(1)≈2.7x limits each order's contribution and keeps pg stable.
            log_ratio = torch.clamp(log_prob - old_lp, -1.0, 1.0)
            ratio = torch.exp(log_ratio)                   # [N_orders]
            pg1 = ratio * adv
            pg2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
            pg_losses.append(-torch.min(pg1, pg2).mean())  # mean across orders
            vf_losses.append(F.mse_loss(value.squeeze(), ret))
            entropies.append(entropy)
            log_ratios.append(log_ratio.detach())

        if not pg_losses:
            return {}

        # Approx KL = E[ratio - 1 - log_ratio] (always ≥ 0, = 0 when ratio = 1).
        # Returned so the epoch loop can stop early if the policy has drifted too
        # far from the collection-time policy (target_kl guard in update()).
        all_lr = torch.cat(log_ratios)
        approx_kl = (torch.exp(all_lr) - 1 - all_lr).mean().item()

        pg = torch.stack(pg_losses).mean()
        vf = torch.stack(vf_losses).mean()
        ent = torch.stack(entropies).mean()
        loss = pg + self.vf_coef * vf - self.ent_coef * ent

        # Skip the update if the loss is non-finite (NaN or Inf).  A single
        # non-finite gradient step permanently corrupts model weights; it is
        # always safer to discard the batch and continue.
        if not loss.isfinite():
            return {}

        self.optimizer.zero_grad()
        loss.backward()

        # Check for Inf/NaN gradients before clip_grad_norm_.
        # If any gradient is Inf, clip_grad_norm_ computes
        #   clip_coef = max_norm / total_norm = max_norm / Inf = 0
        # and then applies param.grad *= clip_coef, giving Inf * 0 = NaN
        # (IEEE 754), which permanently corrupts every model weight.
        # Zeroing and skipping is the safe option.
        has_bad_grad = any(
            p.grad is not None and not p.grad.isfinite().all()
            for p in self.agent.model.parameters()
        )
        if has_bad_grad:
            self.optimizer.zero_grad()
            return {}

        torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {"loss": loss.item(), "pg": pg.item(), "vf": vf.item(), "ent": ent.item(),
                "approx_kl": approx_kl}

    def save_checkpoint(self, path: str | Path) -> None:
        torch.save({
            "model": self.agent.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self._step,
        }, path)

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.agent.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._step = ckpt.get("step", 0)

    def enable_tensorboard(self, log_dir: str = "runs/strategist") -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir)
        except ImportError:
            print("TensorBoard not available.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evaluate(agent: StrategistAgent, n_games: int = EVAL_GAMES, max_turns: int = MAX_TURNS) -> float:
    """Quick win-rate check against GreedyAgent.

    Uses MAX_TURNS (same as training) so that win-rate measurements are
    comparable to what the agent experiences during training.  Using a shorter
    max_turns (e.g. 100) would undercount wins and give misleading Phase A
    exit signals since many games are still in progress at turn 100.
    """
    from ..evaluation.eval_agent import evaluate_agent_vs_greedy

    agent.model.eval()
    result = evaluate_agent_vs_greedy(agent, n_games=n_games, max_turns=max_turns)
    return result.win_rate


def _make_greedy_pool(n: int = 3) -> list[GreedyAgent]:
    """Return a small set of GreedyAgent opponents with varied weight profiles.

    The first agent uses default weights; subsequent agents randomly perturb
    each weight by ±20% so the pool is genuinely diverse rather than n copies
    of the same opponent.
    """
    import random as _rnd

    from ..agents.greedy_agent import DEFAULT_WEIGHTS as GW
    agents = [GreedyAgent(weights=list(GW))]
    rng = _rnd.Random(0)
    for _ in range(n - 1):
        perturbed = [w * rng.uniform(0.8, 1.2) for w in GW]
        agents.append(GreedyAgent(weights=perturbed))
    return agents


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    run_dir: str = "runs/strategist",
    device: str = "cpu",
    n_episodes: int = N_EPISODES,
    n_workers: int = 1,
    total_iters: int = TOTAL_ITERS,
    resume: str | None = None,
    weights: str | None = None,
    skip_bc: bool = False,
    skip_phase_a: bool = False,
    seed: int = 42,
) -> StrategistAgent:
    """
    Full training loop:
      Phase 0 — Behavioural Cloning: imitate GreedyAgent (warm-start)
      Phase A — PPO vs GreedyAgent: fine-tune beyond greedy
      Phase B/C — League self-play: improve via diverse opponents

    Args:
        n_workers: Parallel episode workers (1 = serial).
        skip_bc:   Skip Phase 0 (BC warm-start). Use if resuming training.

    Returns the trained StrategistAgent.
    """
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device)

    # Build model and agent
    model = StrategistGNN(
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        dropout=DROPOUT,
    )
    agent = StrategistAgent(model=model, max_turns=MAX_TURNS, deterministic=False, device=dev)

    trainer = StrategistTrainer(agent=agent, device=dev)
    trainer.enable_tensorboard(str(run_path / "tb"))

    # Build League
    league_cfg = LeagueConfig(
        self_play_prob=0.5,
        snapshot_prob=0.3,
        greedy_prob=0.2,
        max_snapshots=20,
        snapshot_interval=SNAPSHOT_EVERY,
    )
    league = League(learner=agent, config=league_cfg, seed=seed)
    for g in _make_greedy_pool():
        league.add_greedy(g)

    if resume:
        trainer.load_checkpoint(resume)
        print(f"Resumed from {resume} (step {trainer._step})")
        skip_bc = True   # never re-run BC when resuming

    if weights:
        # Load raw model weights (agent.save() format — state_dict only).
        # Used to warm-start from a BC checkpoint without resuming optimizer state.
        agent.load(weights)
        print(f"Loaded weights from {weights}")
        skip_bc = True   # weights already include BC training

    log_path = run_path / "training_log.jsonl"

    # ------------------------------------------------------------------
    # Phase 0: Behavioural Cloning warm-start
    # ------------------------------------------------------------------
    if not skip_bc:
        print("\n=== Phase 0: Behavioural Cloning warm-start ===")
        print(f"  Collecting {BC_GAMES} greedy games → {BC_EPOCHS} training epochs")
        bc_train(
            agent=agent,
            n_games=BC_GAMES,
            n_epochs=BC_EPOCHS,
            device=dev,
            seed=seed,
            verbose=True,
        )
        bc_ckpt = run_path / "ckpt_bc.pt"
        agent.save(bc_ckpt)
        print(f"  BC checkpoint: {bc_ckpt}")
        # Quick sanity-check win rate after BC
        bc_wr = _evaluate(agent)
        print(f"  Win rate after BC: {bc_wr*100:.1f}%  (expect ~16% if imitation worked)")
        with open(log_path, "a") as f:
            f.write(json.dumps({"phase": "BC", "win_rate": bc_wr}) + "\n")

    # ------------------------------------------------------------------
    # Phase A: Bootstrap vs Greedy
    # ------------------------------------------------------------------
    phase_a_done = skip_phase_a
    if not skip_phase_a:
        print("\n=== Phase A: Bootstrap vs GreedyAgent ===", flush=True)
        greedy_only_league = League(learner=agent, seed=seed)
        for g in _make_greedy_pool(n=5):
            greedy_only_league.add_greedy(g)
        # Override mixing: 100% greedy opponents
        greedy_only_league.config.self_play_prob = 0.0
        greedy_only_league.config.snapshot_prob = 0.0
        greedy_only_league.config.greedy_prob = 1.0

        CollectorCls = ParallelStrategistCollector if n_workers > 1 else StrategistCollector
        collector_a_kwargs: dict = dict(
            agent=agent,
            league=greedy_only_league,
            n_episodes=n_episodes,
            max_turns=MAX_TURNS,
            seed=seed,
        )
        if n_workers > 1:
            collector_a_kwargs["n_workers"] = n_workers
        collector_a = CollectorCls(**collector_a_kwargs)

        for it in range(PHASE_A_ITERS):
            print(f"[A] iter={it:03d}  collecting {n_episodes} episodes…", flush=True)
            t0 = time.time()
            buf = collector_a.collect()
            metrics = trainer.update(buf)
            elapsed = time.time() - t0

            evaluated = it % EVAL_EVERY == 0
            win_rate = _evaluate(agent) if evaluated else None

            row = {"phase": "A", "iter": it, "elapsed": round(elapsed, 1),
                   "n_transitions": len(buf), **metrics,
                   "win_rate": win_rate}
            win_str = f"{win_rate*100:.1f}%" if win_rate is not None else "---"
            print(
                f"[A] iter={it:03d}  "
                f"loss={metrics.get('loss', 0):+.4f}  "
                f"pg={metrics.get('pg_loss', 0):+.4f}  "
                f"vf={metrics.get('vf_loss', 0):.4f}  "
                f"ent={metrics.get('entropy', 0):.4f}  "
                f"trans={len(buf):4d}  "
                f"win%={win_str}  ({elapsed:.1f}s)",
                flush=True,
            )
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")

            if it % SNAPSHOT_EVERY == 0:
                ckpt = run_path / f"ckpt_A_{it:04d}.pt"
                trainer.save_checkpoint(ckpt)

            if win_rate is not None and win_rate >= PHASE_A_THRESHOLD:
                print(f"  → Phase A complete (win rate {win_rate:.2%} ≥ {PHASE_A_THRESHOLD:.2%})", flush=True)
                phase_a_done = True
                break

        if not phase_a_done:
            print("  → Phase A complete (max iters reached)")

    # ------------------------------------------------------------------
    # Phase B + C: League self-play
    # ------------------------------------------------------------------
    print("\n=== Phase B/C: League Self-Play ===", flush=True)
    league.snapshot()  # seed pool with current (post-Phase-A) policy

    CollectorCls = ParallelStrategistCollector if n_workers > 1 else StrategistCollector
    collector_kwargs: dict = dict(
        agent=agent,
        league=league,
        n_episodes=n_episodes,
        max_turns=MAX_TURNS,
        seed=seed + 1000,
    )
    if n_workers > 1:
        collector_kwargs["n_workers"] = n_workers
    collector = CollectorCls(**collector_kwargs)

    for it in range(total_iters):
        # Phase C starts at PHASE_B_ITERS: switch to a lower entropy coefficient
        # to encourage more decisive play once the policy is well-established.
        in_phase_c = it >= PHASE_B_ITERS
        if in_phase_c and trainer.ent_coef != ENT_COEF_C:
            trainer.ent_coef = ENT_COEF_C
            print(f"  → Phase C: entropy coef reduced to {ENT_COEF_C}", flush=True)
        phase_label = "C" if in_phase_c else "B"

        print(f"[{phase_label}] iter={it:04d}  collecting {n_episodes} episodes…", flush=True)
        t0 = time.time()
        buf = collector.collect()
        metrics = trainer.update(buf)
        league.on_update()
        elapsed = time.time() - t0

        evaluated = it % EVAL_EVERY == 0
        win_rate = _evaluate(agent) if evaluated else None

        row = {"phase": phase_label, "iter": it, "elapsed": round(elapsed, 1),
               "n_transitions": len(buf), **metrics,
               "win_rate": win_rate, **league.stats()}
        win_str = f"{win_rate*100:.1f}%" if win_rate is not None else "---"
        print(
            f"[{phase_label}] iter={it:04d}  "
            f"loss={metrics.get('loss', 0):+.4f}  "
            f"pg={metrics.get('pg_loss', 0):+.4f}  "
            f"vf={metrics.get('vf_loss', 0):.4f}  "
            f"ent={metrics.get('entropy', 0):.4f}  "
            f"trans={len(buf):4d}  "
            f"win%={win_str}  snaps={league.stats()['n_snapshots']}  ({elapsed:.1f}s)",
            flush=True,
        )
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        if it % SNAPSHOT_EVERY == 0:
            ckpt = run_path / f"ckpt_{phase_label}_{it:04d}.pt"
            trainer.save_checkpoint(ckpt)
            print(f"  → Checkpoint saved: {ckpt}", flush=True)

    # Final save
    final_ckpt = run_path / "ckpt_final.pt"
    trainer.save_checkpoint(final_ckpt)
    print(f"\nTraining complete. Final checkpoint: {final_ckpt}")

    return agent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train StrategistGNN via league PPO")
    p.add_argument("--run-dir",      default="runs/strategist", help="Output directory")
    p.add_argument("--device",       default="cpu",             help="Torch device (cpu/cuda)")
    p.add_argument("--n-episodes",   type=int, default=N_EPISODES)
    p.add_argument("--n-workers",    type=int, default=1,
                   help="Parallel episode workers (1=serial, 0=auto=cpu_count)")
    p.add_argument("--total-iters",  type=int, default=TOTAL_ITERS)
    p.add_argument("--resume",       default=None,              help="Resume from trainer checkpoint (model + optimizer + step)")
    p.add_argument("--weights",      default=None,              help="Load model weights only (agent.save() format, e.g. ckpt_bc.pt)")
    p.add_argument("--skip-bc",      action="store_true",       help="Skip behavioural cloning warm-start")
    p.add_argument("--skip-phase-a", action="store_true",       help="Skip PPO vs greedy bootstrap")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    import os as _os
    n_workers = args.n_workers if args.n_workers > 0 else (_os.cpu_count() or 1)
    train(
        run_dir=args.run_dir,
        device=args.device,
        n_episodes=args.n_episodes,
        n_workers=n_workers,
        total_iters=args.total_iters,
        resume=args.resume,
        weights=args.weights,
        skip_bc=args.skip_bc,
        skip_phase_a=args.skip_phase_a,
        seed=args.seed,
    )
