"""
Rollout buffer for PPO training.

Stores (obs, action, log_prob, value, reward, done) tuples collected
during self-play. After collection, compute_returns() fills in
advantage estimates and discounted returns.

Design:
  - Per-player buffers: each player's trajectory is stored separately
    since they act at different time steps.
  - The buffer stores PyG Data objects (raw encoded boards) to avoid
    re-encoding during the update step.
  - GAE (Generalised Advantage Estimation) is used for variance reduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from torch_geometric.data import Data


@dataclass
class Transition:
    """One player-step tuple."""
    obs: Data            # encoded board state
    acting_mask: Tensor    # [N] bool — source tiles the player owns
    chosen_edges: Tensor   # [K] edge indices selected
    chosen_fractions: Tensor  # [K] fraction values sampled
    log_prob: Tensor       # [K] per-action log prob (or scalar sum)
    value: Tensor          # scalar value estimate
    reward: float
    done: bool


class RolloutBuffer:
    """
    Stores a fixed-horizon trajectory for one player.

    Args:
        gamma:   Discount factor.
        lam:     GAE lambda.
    """

    def __init__(self, gamma: float = 0.99, lam: float = 0.95) -> None:
        self.gamma = gamma
        self.lam = lam
        self._transitions: list[Transition] = []

    def add(self, transition: Transition) -> None:
        self._transitions.append(transition)

    def __len__(self) -> int:
        return len(self._transitions)

    def clear(self) -> None:
        self._transitions.clear()

    def compute_returns(self) -> None:
        """
        Compute GAE advantages and discounted returns in-place.

        Must be called before iterating the buffer for training.
        Stores .advantage and .return_ on each Transition.
        """
        n = len(self._transitions)
        if n == 0:
            return

        advantages = torch.zeros(n)
        gae = 0.0

        for t in reversed(range(n)):
            tr = self._transitions[t]
            reward = tr.reward
            done = float(tr.done)
            value = tr.value.item()

            next_value = self._transitions[t + 1].value.item() if t + 1 < n else 0.0

            delta = reward + self.gamma * next_value * (1 - done) - value
            gae = delta + self.gamma * self.lam * (1 - done) * gae
            advantages[t] = gae

        returns = advantages + torch.tensor(
            [tr.value.item() for tr in self._transitions]
        )

        for t, tr in enumerate(self._transitions):
            tr.advantage = advantages[t]    # type: ignore[attr-defined]
            tr.return_ = returns[t]         # type: ignore[attr-defined]

    def iter_minibatches(self, batch_size: int):
        """
        Yield shuffled mini-batches of Transitions.

        Args:
            batch_size: Number of transitions per mini-batch.

        Yields:
            list[Transition]
        """
        indices = torch.randperm(len(self._transitions))
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            yield [self._transitions[i] for i in batch_idx]
