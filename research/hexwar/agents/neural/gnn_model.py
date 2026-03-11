"""
Graph Neural Network policy/value model for HexWar.

Architecture:
  1. Node encoder MLP: projects raw node features → hidden_dim
  2. Global encoder MLP: projects global features → hidden_dim
  3. L × GATv2Conv message-passing layers (with skip connections)
  4. Per-edge action head: for each directed edge (from_tile → to_tile),
     predicts a (logit, fraction) pair.
       - logit: whether to issue this move at all
       - fraction: Beta distribution parameters (α, β) for fraction of
                   units to send, in (0, 1]
  5. Value head: graph-level value estimate for the acting player.

Action space design:
  The output is over ALL directed edges in the board graph. For each edge
  (u → v) where u is owned by the acting player, the agent outputs:
    - move_logit: pre-softmax logit (action selection)
    - alpha, beta: Beta distribution params for fraction (sampled at train
      time, argmax at eval time)

  During inference, the top-K edges are selected and fraction is sampled
  from Beta(alpha, beta). The final order uses requested_units = round(
  fraction * available_units).

  Masking: edges where `from_tile` is not owned by the acting player are
  masked to -inf before softmax, so the agent can only move its own units.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    pass

from .encoder import N_EDGE_FEATURES, N_GLOBAL_FEATURES, node_feature_dim


class HexWarGNN(nn.Module):
    """
    GNN policy + value network with optional frame-stacked temporal input.

    Args:
        hidden_dim:   Dimension of node embeddings throughout the GNN.
        n_layers:     Number of GATv2Conv message-passing layers.
        n_heads:      Number of attention heads per GATv2Conv layer.
        dropout:      Dropout probability applied after each GNN layer.
        history_k:    Number of historical frames to stack (1 = no history).
                      Node input dim = 18 + (k-1)*4.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        history_k: int = 1,
    ) -> None:
        super().__init__()

        try:
            from torch_geometric.nn import GATv2Conv
        except ImportError as e:
            raise ImportError(
                "torch_geometric not installed. Run: pip install torch_geometric"
            ) from e

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.history_k = history_k
        input_dim = node_feature_dim(history_k)

        # Node feature projection
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Global feature projection
        self.global_encoder = nn.Sequential(
            nn.Linear(N_GLOBAL_FEATURES, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # GATv2Conv layers with skip connections
        head_dim = hidden_dim // n_heads
        self.conv_layers = nn.ModuleList([
            GATv2Conv(
                in_channels=hidden_dim,
                out_channels=head_dim,
                heads=n_heads,
                edge_dim=N_EDGE_FEATURES,
                concat=True,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])

        # Edge action head — consumes concat(src_emb, dst_emb, edge_attr)
        edge_in = hidden_dim * 2 + N_EDGE_FEATURES
        self.move_head = nn.Sequential(
            nn.Linear(edge_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),   # move logit
        )
        self.fraction_head = nn.Sequential(
            nn.Linear(edge_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),   # log(alpha), log(beta) for Beta dist
        )

        # Value head — mean-pool over acting player's nodes
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: Tensor,           # [N, N_NODE_FEATURES]
        edge_index: Tensor,  # [2, E]
        edge_attr: Tensor,   # [E, N_EDGE_FEATURES]
        u: Tensor,           # [B, N_GLOBAL_FEATURES] or [N_GLOBAL_FEATURES]
        acting_mask: Tensor, # [N] bool — True for tiles owned by acting player
        batch: Tensor | None = None,  # [N] batch assignment for PyG batching
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass.

        Returns:
            move_logits:   [E] — raw logits for each directed edge
            alpha:         [E] — Beta distribution alpha (> 0)
            beta:          [E] — Beta distribution beta (> 0)
            value:         [B] — scalar value per graph in batch
        """
        # Encode nodes
        h = self.node_encoder(x)

        # Encode global and broadcast to nodes
        if u.dim() == 1:
            u = u.unsqueeze(0)  # [1, D]
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        u_emb = self.global_encoder(u)         # [B, D]
        h = h + u_emb[batch]                   # broadcast to nodes

        # Message passing
        for conv, norm in zip(self.conv_layers, self.layer_norms):
            h_new = conv(h, edge_index, edge_attr)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = norm(h + h_new)  # skip connection

        # Edge features for action heads
        src, dst = edge_index
        edge_emb = torch.cat([h[src], h[dst], edge_attr], dim=-1)  # [E, 2D+ef]

        move_logits = self.move_head(edge_emb).squeeze(-1)  # [E]

        frac_params = self.fraction_head(edge_emb)           # [E, 2]
        alpha = F.softplus(frac_params[:, 0]) + 1e-4         # ensure > 0
        beta  = F.softplus(frac_params[:, 1]) + 1e-4

        # Value: concat mean over all nodes + mean over acting-player nodes
        from torch_geometric.nn import global_mean_pool
        global_pool = global_mean_pool(h, batch)              # [B, D]

        acting_h = h.clone()
        acting_h[~acting_mask] = 0.0
        acting_count = acting_mask.float().unsqueeze(-1)
        # Safe mean — avoid division by zero
        acting_sum = global_mean_pool(acting_h, batch)
        acting_cnt_by_batch = global_mean_pool(acting_count, batch).clamp(min=1e-6)
        acting_pool = acting_sum / acting_cnt_by_batch        # [B, D]

        value_input = torch.cat([global_pool, acting_pool], dim=-1)  # [B, 2D]
        value = self.value_head(value_input).squeeze(-1)              # [B]

        return move_logits, alpha, beta, value
