"""
Strategist GNN — next-generation policy/value model for HexWar.

Architecture improvements over HexWarGNN:

1. Per-tile GRU hidden state (replaces frame stacking)
   - Frame stacking (K=5) only sees the last 5 turns.
   - A GRUCell applied per-tile accumulates information over ALL past turns.
   - Enables detecting slow buildups, evacuations, and strategic patterns
     that span 20–50 turns — invisible to frame-stacked models.
   - The hidden state [N, hidden_dim] is maintained by StrategistAgent between
     turns and passed in on each forward call.

2. Global self-attention after local message passing
   - GATv2Conv with L=4 layers reaches at most 4 hops.
   - On a typical 100-tile board with diameter ~12–15, large parts of the
     board are effectively invisible to any single tile's decision.
   - One MultiheadAttention layer over ALL tiles (O(N²), N~100 → negligible)
     gives every tile full visibility of global board state before acting.
   - This enables reasoning like "the far flank is collapsing; I should push
     elsewhere" — impossible with local message passing alone.

3. Phase input
   - Explicit normalised turn counter broadcast to all nodes.
   - Allows the policy to adapt strategy to early/mid/late game automatically.

Action space: same as HexWarGNN (per-edge logit + Beta fraction).
Value head:   same as HexWarGNN (global + acting-player pool).

Forward signature change vs HexWarGNN:
  - Does NOT accept history_k / stacked frames.
  - Accepts `h_tiles` [N, hidden_dim] (previous GRU state) and returns
    `h_tiles_new` [N, hidden_dim] as a fifth output.
  - Input `x` is always the CURRENT frame only (N_NODE_FEATURES = 18 dims).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .encoder import N_EDGE_FEATURES, N_GLOBAL_FEATURES, N_NODE_FEATURES


class StrategistGNN(nn.Module):
    """
    GNN policy + value network with per-tile GRU temporal state and global
    self-attention.

    Args:
        hidden_dim:  Dimension of node embeddings throughout the network.
        n_layers:    Number of GATv2Conv message-passing layers.
        n_heads:     Attention heads for both GATv2Conv and global attention.
        dropout:     Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
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

        # Node feature projection (current frame only — no stacking)
        # +1 for normalised turn counter appended to each node
        self.node_encoder = nn.Sequential(
            nn.Linear(N_NODE_FEATURES + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Global feature projection
        self.global_encoder = nn.Sequential(
            nn.Linear(N_GLOBAL_FEATURES, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Per-tile GRU cell — temporal memory across turns
        # GRUCell: input [N, hidden_dim], hidden [N, hidden_dim] → [N, hidden_dim]
        self.gru_cell = nn.GRUCell(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
        )

        # Local message passing (GATv2Conv)
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

        # Global self-attention — full board visibility
        # Each tile attends to every other tile after local message passing.
        self.global_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_global = nn.LayerNorm(hidden_dim)

        # Edge action head — same structure as HexWarGNN
        edge_in = hidden_dim * 2 + N_EDGE_FEATURES
        self.move_head = nn.Sequential(
            nn.Linear(edge_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.fraction_head = nn.Sequential(
            nn.Linear(edge_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: Tensor,            # [N, N_NODE_FEATURES] — CURRENT frame only
        edge_index: Tensor,   # [2, E]
        edge_attr: Tensor,    # [E, N_EDGE_FEATURES]
        u: Tensor,            # [N_GLOBAL_FEATURES]
        acting_mask: Tensor,  # [N] bool
        h_tiles: Tensor | None = None,   # [N, hidden_dim] — previous GRU state
        turn_frac: float = 0.0,          # normalised turn (0–1) for phase awareness
        batch: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass.

        Returns:
            move_logits:  [E]
            alpha:        [E]  Beta distribution alpha  (> 0)
            beta:         [E]  Beta distribution beta   (> 0)
            value:        [B]  scalar value per graph
            h_tiles_new:  [N, hidden_dim]  updated GRU hidden state
        """
        N = x.size(0)
        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=x.device)

        # Append normalised turn counter to node features (phase awareness)
        turn_feat = torch.full((N, 1), turn_frac, device=x.device, dtype=x.dtype)
        x_aug = torch.cat([x, turn_feat], dim=-1)  # [N, N_NODE_FEATURES + 1]

        # 1. Encode nodes
        h = self.node_encoder(x_aug)   # [N, hidden_dim]

        # 2. Broadcast global features
        if u.dim() == 1:
            u = u.unsqueeze(0)
        u_emb = self.global_encoder(u)   # [B, hidden_dim]
        h = h + u_emb[batch]             # [N, hidden_dim]

        # 3. Per-tile GRU update (temporal memory)
        if h_tiles is None:
            h_tiles = torch.zeros(N, self.hidden_dim, device=x.device, dtype=x.dtype)
        h = self.gru_cell(h, h_tiles)    # [N, hidden_dim] — new hidden state
        h_tiles_new = h.detach()         # detach for storage; gradients still flow through h

        # 4. Local message passing (GATv2Conv with skip connections)
        for conv, norm in zip(self.conv_layers, self.layer_norms):
            h_mp = conv(h, edge_index, edge_attr)
            h_mp = F.dropout(h_mp, p=self.dropout, training=self.training)
            h = norm(h + h_mp)

        # 5. Global self-attention — full board visibility
        # All tiles attend to each other, bypassing the hop-distance limit.
        h_seq = h.unsqueeze(0)                          # [1, N, hidden_dim]
        h_attn, _ = self.global_attn(h_seq, h_seq, h_seq)
        h = self.norm_global(h + h_attn.squeeze(0))     # [N, hidden_dim]

        # 6. Edge action heads
        src, dst = edge_index
        edge_emb = torch.cat([h[src], h[dst], edge_attr], dim=-1)  # [E, 2D+ef]

        move_logits = self.move_head(edge_emb).squeeze(-1)          # [E]

        frac_params = self.fraction_head(edge_emb)                  # [E, 2]
        alpha = F.softplus(frac_params[:, 0]) + 1e-4
        beta  = F.softplus(frac_params[:, 1]) + 1e-4

        # 7. Value head (global pool + acting-player pool)
        from torch_geometric.nn import global_mean_pool
        global_pool = global_mean_pool(h, batch)                    # [B, D]

        acting_h = h.clone()
        acting_h[~acting_mask] = 0.0
        acting_count = acting_mask.float().unsqueeze(-1)
        acting_sum = global_mean_pool(acting_h, batch)
        acting_cnt = global_mean_pool(acting_count, batch).clamp(min=1e-6)
        acting_pool = acting_sum / acting_cnt                       # [B, D]

        value_input = torch.cat([global_pool, acting_pool], dim=-1) # [B, 2D]
        value = self.value_head(value_input).squeeze(-1)            # [B]

        return move_logits, alpha, beta, value, h_tiles_new
