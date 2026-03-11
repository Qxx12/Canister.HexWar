"""
PPO agent — wraps HexWarGNN for inference and rollout collection.

At inference time:
  1. Encode the board with encoder.encode_board().
  2. Forward through HexWarGNN.
  3. Mask edges whose source tile is not owned by the acting player.
  4. Sample top-K moves (K=1 per source tile) using Categorical over
     masked logits. K is capped at one move per source tile.
  5. For each selected move, sample fraction from Beta(alpha, beta).
  6. Convert (edge, fraction) → MovementOrder.

At train time the same forward pass is used; the PPOTrainer collects
(obs, action, log_prob, value, reward, done) tuples into a RolloutBuffer.

Deterministic mode (training=False):
  - Uses argmax over move logits (no sampling).
  - fraction = alpha / (alpha + beta)  (mode of Beta distribution).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Beta

if TYPE_CHECKING:
    from torch_geometric.data import Data

from ...engine.types import Board, MovementOrder, OrderMap, Player, PlayerStats
from ..base_agent import BaseAgent
from .encoder import encode_board_with_history
from .gnn_model import HexWarGNN
from .history_buffer import HistoryBuffer


class PPOAgent(BaseAgent):
    """
    Neural PPO agent using a GNN policy with frame-stacked temporal input.

    Each agent instance maintains its own HistoryBuffer so it accumulates
    board snapshots across turns within a game. Call reset() between games.

    Args:
        model:         A HexWarGNN instance.
        history_k:     Number of frames to stack (1 = current frame only,
                       5 = current + 4 history frames).
        deterministic: If True, uses greedy action selection (no sampling).
        device:        Torch device.
    """

    def __init__(
        self,
        model: HexWarGNN | None = None,
        history_k: int = 5,
        deterministic: bool = False,
        device: str | torch.device = "cpu",
        **model_kwargs,
    ) -> None:
        self.device = torch.device(device)
        self.history_k = history_k
        self.model = model if model is not None else HexWarGNN(history_k=history_k, **model_kwargs)
        self.model = self.model.to(self.device)
        self._history = HistoryBuffer(k=history_k)
        self.deterministic = deterministic

    # ------------------------------------------------------------------
    # AgentFn protocol
    # ------------------------------------------------------------------

    def reset(self, initial_board: Board | None = None) -> None:
        """Reset history buffer between games."""
        if initial_board is not None:
            self._history.reset(initial_board)
        else:
            self._history._frames.clear()

    def __call__(
        self,
        board: Board,
        player_id: str,
        players: list[Player],
        stats: dict[str, PlayerStats],
    ) -> OrderMap:
        """Produce orders for one turn. Pushes board to history buffer."""
        # Initialise buffer on first call of the game
        if len(self._history) == 0:
            self._history.reset(board)

        with torch.no_grad():
            orders = self._act(board, player_id)

        # Record this board state for the next turn's history
        self._history.push(board)
        return orders

    def _act(self, board: Board, player_id: str) -> OrderMap:
        frames = self._history.get_frames()
        # Replace the last (oldest replicated) frame with current board
        frames[-1] = board
        data = encode_board_with_history(frames, player_id)
        data = data.to(self.device)

        node_keys: list[str] = data.node_keys  # type: ignore[assignment]

        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        u = data.u

        # Acting mask: True for tiles owned by the acting player WITH units
        acting_mask = torch.tensor(
            [
                board[k].owner == player_id and board[k].units > 0
                for k in node_keys
            ],
            dtype=torch.bool,
            device=self.device,
        )

        move_logits, alpha, beta, _value = self.model(
            x, edge_index, edge_attr, u, acting_mask, batch=None
        )

        # Mask edges where source tile is not owned/active
        src = edge_index[0]
        valid_src = acting_mask[src]
        move_logits = move_logits.masked_fill(~valid_src, -1e9)

        # Select one best move per source tile
        orders = self._select_orders(
            move_logits, alpha, beta, edge_index, node_keys, board, player_id
        )
        return orders

    def _select_orders(
        self,
        move_logits: Tensor,
        alpha: Tensor,
        beta: Tensor,
        edge_index: Tensor,
        node_keys: list[str],
        board: Board,
        player_id: str,
    ) -> OrderMap:
        orders: OrderMap = {}
        src, dst = edge_index

        # For each source node, pick the highest-logit outgoing edge
        src_to_best: dict[int, int] = {}  # src_idx → edge_idx
        for edge_idx in range(src.size(0)):
            s = src[edge_idx].item()
            if s not in src_to_best or move_logits[edge_idx].item() > move_logits[src_to_best[s]].item():
                src_to_best[s] = edge_idx

        for src_idx, edge_idx in src_to_best.items():
            from_key = node_keys[src_idx]
            to_key = node_keys[dst[edge_idx].item()]
            from_tile = board[from_key]

            if from_tile.owner != player_id or from_tile.units == 0:
                continue

            # Only issue move if logit is not masked
            if move_logits[edge_idx].item() < -1e8:
                continue

            # Sample or compute fraction
            a = alpha[edge_idx].item()
            b = beta[edge_idx].item()
            if self.deterministic:
                # Mode of Beta(a, b) — clamp for a,b ≤ 1 edge case
                frac = (a - 1) / (a + b - 2) if a > 1 and b > 1 else a / (a + b)
            else:
                frac = Beta(
                    torch.tensor(a), torch.tensor(b)
                ).sample().item()

            units_to_send = max(1, round(frac * from_tile.units))
            orders[from_key] = MovementOrder(
                from_key=from_key,
                to_key=to_key,
                requested_units=units_to_send,
            )

        return orders

    # ------------------------------------------------------------------
    # Log probability (used by PPOTrainer)
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        data: Data,
        acting_mask: Tensor,
        chosen_edges: Tensor,      # [N_orders] indices into edge_index
        chosen_fractions: Tensor,  # [N_orders] fractions in (0, 1]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute log_prob, entropy, and value for a batch of stored actions.

        Used by PPOTrainer during the policy update step.

        Returns:
            log_prob: [N_orders] sum of log P(edge) + log P(fraction)
            entropy:  scalar entropy estimate
            value:    [1] value estimate
        """
        move_logits, alpha, beta, value = self.model(
            data.x, data.edge_index, data.edge_attr, data.u, acting_mask
        )

        src = data.edge_index[0]
        valid_src = acting_mask[src]
        move_logits = move_logits.masked_fill(~valid_src, -1e9)

        # Edge selection log prob (Categorical over valid edges)
        log_probs_edge = F.log_softmax(move_logits, dim=0)
        lp_edge = log_probs_edge[chosen_edges]

        # Fraction log prob (Beta)
        dist = Beta(alpha[chosen_edges], beta[chosen_edges])
        lp_frac = dist.log_prob(chosen_fractions.clamp(1e-6, 1 - 1e-6))

        log_prob = lp_edge + lp_frac

        # Entropy: edge entropy + mean fraction entropy
        edge_entropy = -(F.softmax(move_logits, dim=0) * F.log_softmax(move_logits, dim=0)).sum()
        frac_entropy = Beta(alpha, beta).entropy().mean()
        entropy = edge_entropy + frac_entropy

        return log_prob, entropy, value

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
