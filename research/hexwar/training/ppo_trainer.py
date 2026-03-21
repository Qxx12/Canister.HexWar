"""
PPO trainer — policy gradient update using clipped surrogate objective.

Implements the standard PPO-clip algorithm (Schulman et al. 2017):
  L_CLIP = E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)]
  L_VF   = MSE(V(s), R_t)
  L_ENT  = entropy bonus (encourages exploration)

  Total loss = -L_CLIP + c1 * L_VF - c2 * L_ENT

Training loop::

    trainer = PPOTrainer(agent, lr=3e-4)
    for iteration in range(N_ITERATIONS):
        buffers = collector.collect()
        trainer.update(buffers)
        trainer.save_checkpoint(f"ckpt_{iteration}.pt")

The trainer handles:
  - Multiple epochs over the collected rollout (PPO reuse).
  - Mini-batch updates to avoid memory overflow.
  - Gradient clipping.
  - TensorBoard logging.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam

from ..agents.neural.ppo_agent import PPOAgent
from .rollout_buffer import RolloutBuffer, Transition


class PPOTrainer:
    """
    Proximal Policy Optimisation trainer for HexWar.

    Args:
        agent:          The PPOAgent whose model will be updated.
        lr:             Learning rate.
        clip_eps:       PPO clipping epsilon.
        vf_coef:        Value function loss coefficient.
        ent_coef:       Entropy bonus coefficient.
        max_grad_norm:  Gradient clipping norm.
        n_epochs:       Number of passes over collected data per update.
        batch_size:     Mini-batch size (transitions).
        device:         Torch device.
    """

    def __init__(
        self,
        agent: PPOAgent,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 32,
        device: str | torch.device = "cpu",
    ) -> None:
        self.agent = agent
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.agent.model = self.agent.model.to(self.device)

        self.optimizer = Adam(agent.model.parameters(), lr=lr)
        self._step = 0

        # Optional TensorBoard writer (initialised lazily)
        self._writer = None

    def update(self, buffers: dict[str, RolloutBuffer]) -> dict[str, float]:
        """
        Perform PPO update using all collected transitions.

        Args:
            buffers: Per-player RolloutBuffers (returns already computed).

        Returns:
            Dict of mean training metrics for logging.
        """
        # Flatten all player transitions into one pool
        all_transitions: list[Transition] = []
        for buf in buffers.values():
            all_transitions.extend(buf._transitions)

        if not all_transitions:
            return {}

        total_loss = 0.0
        total_pg = 0.0
        total_vf = 0.0
        total_ent = 0.0
        n_updates = 0

        self.agent.model.train()

        for _ in range(self.n_epochs):
            import random
            random.shuffle(all_transitions)
            for start in range(0, len(all_transitions), self.batch_size):
                batch = all_transitions[start : start + self.batch_size]
                metrics = self._update_batch(batch)
                total_loss += metrics["loss"]
                total_pg   += metrics["pg_loss"]
                total_vf   += metrics["vf_loss"]
                total_ent  += metrics["entropy"]
                n_updates  += 1

        self.agent.model.eval()
        self._step += 1

        result = {
            "loss":     total_loss / max(n_updates, 1),
            "pg_loss":  total_pg   / max(n_updates, 1),
            "vf_loss":  total_vf   / max(n_updates, 1),
            "entropy":  total_ent  / max(n_updates, 1),
        }

        if self._writer:
            for k, v in result.items():
                self._writer.add_scalar(f"train/{k}", v, self._step)

        return result

    def _update_batch(self, batch: list[Transition]) -> dict[str, float]:
        """Run one mini-batch update step and return metrics."""
        try:
            from torch_geometric.data import Batch  # noqa: F401
        except ImportError as err:
            raise ImportError("torch_geometric not installed.") from err

        # Normalize advantages across the batch for training stability
        raw_advantages = torch.stack([tr.advantage for tr in batch])
        norm_advantages = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std() + 1e-8)

        pg_losses = []
        vf_losses = []
        entropies = []

        for i, tr in enumerate(batch):
            obs = tr.obs.to(self.device)
            acting_mask = tr.acting_mask.to(self.device)
            chosen_edges = tr.chosen_edges.to(self.device)
            chosen_fracs = tr.chosen_fractions.to(self.device)
            old_log_prob = tr.log_prob.to(self.device)
            ret = tr.return_.to(self.device)
            adv = norm_advantages[i].to(self.device)

            if chosen_edges.numel() == 0:
                continue

            log_prob, entropy, value = self.agent.evaluate_actions(
                obs, acting_mask, chosen_edges, chosen_fracs
            )

            log_prob_sum = log_prob.sum()
            ratio = torch.exp(log_prob_sum - old_log_prob)

            # PPO clip
            pg1 = ratio * adv
            pg2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
            pg_loss = -torch.min(pg1, pg2)

            # Value function loss
            vf_loss = F.mse_loss(value.squeeze(), ret)

            pg_losses.append(pg_loss)
            vf_losses.append(vf_loss)
            entropies.append(entropy)

        if not pg_losses:
            return {"loss": 0.0, "pg_loss": 0.0, "vf_loss": 0.0, "entropy": 0.0}

        pg_loss_mean = torch.stack(pg_losses).mean()
        vf_loss_mean = torch.stack(vf_losses).mean()
        ent_mean = torch.stack(entropies).mean()

        loss = pg_loss_mean + self.vf_coef * vf_loss_mean - self.ent_coef * ent_mean

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "loss":    loss.item(),
            "pg_loss": pg_loss_mean.item(),
            "vf_loss": vf_loss_mean.item(),
            "entropy": ent_mean.item(),
        }

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

    def enable_tensorboard(self, log_dir: str = "runs/ppo") -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir)
        except ImportError:
            print("TensorBoard not available. Install tensorboard to enable logging.")
