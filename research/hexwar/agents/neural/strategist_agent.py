"""
StrategistAgent — inference wrapper for StrategistGNN.

Key differences from PPOAgent / HexWarGNN:
  - Uses encode_board() (current frame only) — no frame stacking.
  - Maintains per-tile GRU hidden state (h_tiles) across turns inside a game.
    The state is stored as a float32 tensor aligned to the current board's node
    ordering.  On board resize (player eliminated) the tensor is rebuilt.
  - Passes turn_frac (normalised turn counter) for phase-aware reasoning.
  - Conflict-aware coordinated action selection:
      1. Sort source tiles by their best-edge logit (most confident first).
      2. Assign each source tile to its highest-logit unclaimed target.
      3. If a target has already been committed to by a higher-confidence source
         with enough units to conquer it, redirect the lower-confidence source to
         its next-best target instead.
    This prevents pointless pile-ons where 3 tiles all send troops to the same
    weak neutral while leaving the flank undefended.
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
from .encoder import encode_board
from .strategist_model import StrategistGNN


class StrategistAgent(BaseAgent):
    """
    Neural agent using StrategistGNN with per-tile temporal GRU state.

    Args:
        model:         A StrategistGNN instance (created with defaults if None).
        max_turns:     Expected game length; used to normalise turn_frac.
        deterministic: Greedy action selection when True; Beta-sample when False.
        device:        Torch device string or object.
    """

    def __init__(
        self,
        model: StrategistGNN | None = None,
        max_turns: int = 200,
        deterministic: bool = False,
        device: str | torch.device = "cpu",
        **model_kwargs,
    ) -> None:
        self.device = torch.device(device)
        self.max_turns = max_turns
        self.deterministic = deterministic

        self.model = model if model is not None else StrategistGNN(**model_kwargs)
        self.model = self.model.to(self.device)

        # Per-tile GRU hidden state: rebuilt each game on reset()
        self._h_tiles: Tensor | None = None   # [N, hidden_dim]
        self._node_keys: list[str] = []        # current alignment of h_tiles rows
        self._turn: int = 0

    # ------------------------------------------------------------------
    # BaseAgent / AgentFn protocol
    # ------------------------------------------------------------------

    def reset(self, initial_board: Board | None = None) -> None:
        """Clear GRU state and turn counter between games."""
        self._h_tiles = None
        self._node_keys = []
        self._turn = 0

    def __call__(
        self,
        board: Board,
        player_id: str,
        players: list[Player],
        stats: dict[str, PlayerStats],
    ) -> OrderMap:
        with torch.no_grad():
            orders = self._act(board, player_id)
        self._turn += 1
        return orders

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _act(self, board: Board, player_id: str) -> OrderMap:
        data = encode_board(board, player_id)
        data = data.to(self.device)

        node_keys: list[str] = data.node_keys  # type: ignore[assignment]
        N = data.x.size(0)

        # Align or rebuild GRU hidden state to current node ordering
        h_tiles = self._get_aligned_h(node_keys, N, data.x.device)

        turn_frac = min(1.0, self._turn / max(1, self.max_turns))

        acting_mask = torch.tensor(
            [board[k].owner == player_id and board[k].units > 0 for k in node_keys],
            dtype=torch.bool,
            device=self.device,
        )

        move_logits, alpha, beta, _value, h_tiles_new = self.model(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.u,
            acting_mask,
            h_tiles=h_tiles,
            turn_frac=turn_frac,
            batch=None,
        )

        # Store updated GRU state (already detached inside model)
        self._h_tiles = h_tiles_new
        self._node_keys = node_keys

        # Mask edges from non-acting tiles
        src = data.edge_index[0]
        valid_src = acting_mask[src]
        move_logits = move_logits.masked_fill(~valid_src, -1e9)

        return self._select_orders_coordinated(
            move_logits, alpha, beta, data.edge_index, node_keys, board, player_id
        )

    # ------------------------------------------------------------------
    # Conflict-aware coordinated action selection
    # ------------------------------------------------------------------

    def _select_orders_coordinated(
        self,
        move_logits: Tensor,
        alpha: Tensor,
        beta: Tensor,
        edge_index: Tensor,
        node_keys: list[str],
        board: Board,
        player_id: str,
    ) -> OrderMap:
        """
        For each acting source tile, pick the best target that isn't already
        claimed by a higher-confidence attacker with sufficient force.

        Algorithm:
          1. For every acting source tile, collect its edges sorted by logit ↓.
          2. Sort sources by their top-logit ↓ (most decisive first).
          3. Greedily assign: each source takes its highest available target.
             A target is "saturated" once committed units ≥ target.units + 1
             (enough to capture).  Subsequent sources skip saturated targets.
        """
        src_t, dst_t = edge_index

        # Group edges per source: {src_idx: [(logit, edge_idx), ...] sorted ↓}
        src_edges: dict[int, list[tuple[float, int]]] = {}
        for eidx in range(src_t.size(0)):
            s = src_t[eidx].item()
            logit = move_logits[eidx].item()
            if logit < -1e8:
                continue  # masked
            if s not in src_edges:
                src_edges[s] = []
            src_edges[s].append((logit, eidx))

        for s in src_edges:
            src_edges[s].sort(key=lambda t: t[0], reverse=True)

        # Sort sources by their best logit (highest confidence first)
        sorted_srcs = sorted(src_edges.keys(), key=lambda s: src_edges[s][0][0], reverse=True)

        committed_units: dict[int, int] = {}   # dst_idx → total units committed
        orders: OrderMap = {}

        for src_idx in sorted_srcs:
            from_key = node_keys[src_idx]
            from_tile = board[from_key]
            if from_tile.owner != player_id or from_tile.units == 0:
                continue

            frac, chosen_eidx = self._pick_edge(
                src_idx, src_edges[src_idx], alpha, beta, dst_t,
                node_keys, board, player_id, committed_units
            )
            if chosen_eidx is None:
                continue

            dst_idx = dst_t[chosen_eidx].item()
            to_key = node_keys[dst_idx]

            units_to_send = max(1, round(frac * from_tile.units))
            committed_units[dst_idx] = committed_units.get(dst_idx, 0) + units_to_send

            orders[from_key] = MovementOrder(
                from_key=from_key,
                to_key=to_key,
                requested_units=units_to_send,
            )

        return orders

    def _pick_edge(
        self,
        src_idx: int,
        candidates: list[tuple[float, int]],  # sorted by logit ↓
        alpha: Tensor,
        beta: Tensor,
        dst_t: Tensor,
        node_keys: list[str],
        board: Board,
        player_id: str,
        committed_units: dict[int, int],
    ) -> tuple[float, int | None]:
        """
        Return (fraction, edge_idx) for the best unsaturated target edge,
        or (0.0, None) if all targets are already saturated.
        """
        for _logit, eidx in candidates:
            dst_idx = dst_t[eidx].item()
            to_key = node_keys[dst_idx]
            to_tile = board[to_key]

            # Compute fraction for this edge
            a = alpha[eidx].item()
            b = beta[eidx].item()
            if self.deterministic:
                frac = (a - 1) / (a + b - 2) if a > 1 and b > 1 else a / (a + b)
            else:
                frac = Beta(torch.tensor(a), torch.tensor(b)).sample().item()

            # Is this target already captured/saturated?
            already_committed = committed_units.get(dst_idx, 0)
            target_is_own = to_tile.owner == player_id

            if target_is_own:
                # Reinforcing own tile — always OK (no saturation concept)
                return frac, eidx

            # Enemy/neutral: saturated if already committed units exceed defense
            target_units = to_tile.units if to_tile.units is not None else 0
            if already_committed > target_units:
                # Pile-on: skip this target, try next best
                continue

            return frac, eidx

        return 0.0, None

    # ------------------------------------------------------------------
    # GRU state alignment
    # ------------------------------------------------------------------

    def _get_aligned_h(
        self, node_keys: list[str], N: int, dev: torch.device
    ) -> Tensor:
        """
        Return h_tiles [N, hidden_dim] aligned to node_keys.

        If the board changed (tiles added/removed after player elimination),
        we rebuild the tensor, copying over rows whose keys still exist.
        """
        hidden_dim = self.model.hidden_dim

        if self._h_tiles is None or self._node_keys != node_keys:
            h_new = torch.zeros(N, hidden_dim, device=dev, dtype=torch.float32)
            if self._h_tiles is not None and self._node_keys:
                # Copy surviving rows
                old_idx = {k: i for i, k in enumerate(self._node_keys)}
                for new_i, key in enumerate(node_keys):
                    if key in old_idx:
                        h_new[new_i] = self._h_tiles[old_idx[key]].to(dev)
            return h_new

        return self._h_tiles.to(dev)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)

    # ------------------------------------------------------------------
    # Training helpers (used by PPOTrainer / StrategistTrainer)
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        data: Data,
        acting_mask: Tensor,
        chosen_edges: Tensor,
        chosen_fractions: Tensor,
        h_tiles: Tensor | None = None,
        turn_frac: float = 0.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute log_prob, entropy, and value for stored actions.

        Returns:
            log_prob: [N_orders]
            entropy:  scalar
            value:    [1]
        """
        move_logits, alpha, beta, value, _ = self.model(
            data.x, data.edge_index, data.edge_attr, data.u,
            acting_mask, h_tiles=h_tiles, turn_frac=turn_frac, batch=None,
        )

        src = data.edge_index[0]
        valid_src = acting_mask[src]
        move_logits = move_logits.masked_fill(~valid_src, -1e9)

        log_probs_edge = F.log_softmax(move_logits, dim=0)
        lp_edge = log_probs_edge[chosen_edges]

        dist = Beta(alpha[chosen_edges], beta[chosen_edges])
        lp_frac = dist.log_prob(chosen_fractions.clamp(1e-6, 1 - 1e-6))

        log_prob = lp_edge + lp_frac

        edge_entropy = -(F.softmax(move_logits, dim=0) * F.log_softmax(move_logits, dim=0)).sum()
        # Compute fraction entropy only over the actually-chosen edges.
        # Using Beta(alpha, beta).entropy().mean() over ALL edges (~600) includes
        # edges the policy never selected and dilutes the entropy signal with
        # arbitrary distributions, producing misleading (often very negative)
        # entropy values that corrupt the entropy bonus term.
        frac_entropy = Beta(alpha[chosen_edges], beta[chosen_edges]).entropy().mean() \
            if chosen_edges.numel() > 0 else torch.tensor(0.0, device=alpha.device)
        entropy = edge_entropy + frac_entropy

        return log_prob, entropy, value
