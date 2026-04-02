"""
Behavioural Cloning (BC) warm-start for StrategistGNN.

Trains the neural network to imitate a GreedyAgent via supervised learning
before PPO fine-tuning begins. This solves the cold-start problem: instead of
starting from random weights (0% win rate), the model begins Phase A already
playing at greedy level (~16% win rate in a 6-player game).

What BC teaches the model:
  - Edge selection: which tile to attack/reinforce from each source tile
    (cross-entropy loss over edges, teacher = greedy's chosen edge)
  - Fraction: how many units to send
    (MSE loss on Beta mean vs greedy's send_fraction)

After BC, PPO fine-tunes the policy to exceed greedy — something BC alone
cannot achieve because greedy is the ceiling of the teacher.

Typical schedule:
  bc_train(agent, n_games=200, n_epochs=10)   # ~5-10 min on CPU
  train(agent=agent, skip_phase_a=False, ...)  # PPO from greedy-level init

Usage::

    python -m hexwar.training.behavioural_cloning
    python -m hexwar.training.behavioural_cloning --games 500 --epochs 20
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam

from ..agents.greedy_agent import DEFAULT_WEIGHTS, GreedyAgent
from ..agents.neural.encoder import encode_board
from ..agents.neural.strategist_agent import StrategistAgent
from ..agents.neural.strategist_model import StrategistGNN
from ..engine.game_engine import HexWarEnv
from ..engine.types import PLAYER_IDS
from ..engine.unit_generator import generate_units


@dataclass
class BCExample:
    """One (board, greedy_action) supervised example."""
    obs: object          # PyG Data — encoded board
    acting_mask: object  # [N] bool
    target_edge: int     # index into edge_index that greedy chose
    target_frac: float   # fraction greedy sent


def collect_bc_data(
    n_games: int = 200,
    max_turns: int = 150,
    seed: int = 0,
    n_workers: int = 1,
) -> list[BCExample]:
    """
    Run all-greedy games and record (board, action) pairs for every player turn.

    Returns a flat list of BCExample — one per greedy player turn.
    """
    examples: list[BCExample] = []

    for game_idx in range(n_games):
        agents = {pid: GreedyAgent(weights=list(DEFAULT_WEIGHTS)) for pid in PLAYER_IDS}
        env = HexWarEnv(agents=agents, seed=seed + game_idx, max_turns=max_turns)

        turn_number = 0
        while env.phase != "end":

            if env.phase == "generateUnits":
                generate_units(env.board, env.stats)
                turn_number += 1
                env.turn.turn_number += 1
                env.turn.active_ai_index = 0
                if env.turn.turn_number > max_turns:
                    env._end_game(winner_id=None)
                    break
                env.phase = "playerTurn"
                continue

            idx = env.turn.active_ai_index
            if idx >= len(env.player_ids):
                env.phase = "generateUnits"
                continue

            pid = env.player_ids[idx]
            env.turn.active_ai_index += 1

            player = next((p for p in env.players if p.id == pid), None)
            if player is None or player.is_eliminated:
                continue

            agent = agents[pid]

            # Ask greedy what it would do
            greedy_orders = agent(env.board, pid, env.players, env.stats)

            if not greedy_orders:
                # Greedy passed — no useful example
                from ..engine.turn_resolver import resolve_player_turn
                result = resolve_player_turn(
                    board=env.board, players=env.players,
                    orders={}, player_id=pid, stats=env.stats,
                )
                env.board = result.board
                env.players = result.players
                env.stats = result.stats
                for p in env.players:
                    if p.is_eliminated and p.id not in env._elimination_order:
                        env._elimination_order.append(p.id)
                if result.winner_id:
                    env._end_game(winner_id=result.winner_id)
                    break
                continue

            # Encode board as the neural net sees it
            obs = encode_board(env.board, pid)
            node_keys: list[str] = obs.node_keys  # type: ignore[assignment]
            key_to_idx = {k: i for i, k in enumerate(node_keys)}

            acting_mask = torch.tensor(
                [env.board[k].owner == pid and env.board[k].units > 0
                 for k in node_keys],
                dtype=torch.bool,
            )

            src_t, dst_t = obs.edge_index

            # For each greedy order, find the matching edge in the graph
            for from_key, order in greedy_orders.items():
                from_idx = key_to_idx.get(from_key)
                to_idx = key_to_idx.get(order.to_key)
                if from_idx is None or to_idx is None:
                    continue

                # Find edge index
                target_edge = None
                for ei in range(src_t.size(0)):
                    if src_t[ei].item() == from_idx and dst_t[ei].item() == to_idx:
                        target_edge = ei
                        break

                if target_edge is None:
                    continue

                # Fraction: units_sent / total_units
                from_tile = env.board[from_key]
                if from_tile.units <= 0:
                    continue
                frac = min(1.0, order.requested_units / from_tile.units)

                examples.append(BCExample(
                    obs=obs,
                    acting_mask=acting_mask,
                    target_edge=target_edge,
                    target_frac=frac,
                ))

            # Resolve the turn with greedy's orders
            from ..engine.turn_resolver import resolve_player_turn
            result = resolve_player_turn(
                board=env.board, players=env.players,
                orders=greedy_orders, player_id=pid, stats=env.stats,
            )
            env.board = result.board
            env.players = result.players
            env.stats = result.stats
            for p in env.players:
                if p.is_eliminated and p.id not in env._elimination_order:
                    env._elimination_order.append(p.id)
            if result.winner_id:
                env._end_game(winner_id=result.winner_id)
                break

    return examples


def bc_train(
    agent: StrategistAgent,
    n_games: int = 200,
    n_epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_turns: int = 150,
    seed: int = 0,
    device: str | torch.device = "cpu",
    verbose: bool = True,
    max_examples: int = 10_000,
) -> None:
    """
    Behavioural cloning warm-start.

    Trains agent.model in-place to imitate GreedyAgent.

    Args:
        agent:      StrategistAgent to warm-start (modified in-place).
        n_games:    Number of all-greedy games to generate training data.
        n_epochs:   Training epochs over the collected dataset.
        lr:         Learning rate (higher than PPO — we want fast imitation).
        batch_size: Examples per gradient step.
        max_turns:  Max turns per game during data collection.
        seed:       RNG seed.
        device:     Training device.
        verbose:    Print per-epoch stats.
    """
    dev = torch.device(device)
    agent.model = agent.model.to(dev)

    if verbose:
        print(f"Collecting BC data from {n_games} greedy games...", flush=True)

    examples = collect_bc_data(n_games=n_games, max_turns=max_turns, seed=seed)

    if not examples:
        print("Warning: no BC examples collected — skipping BC phase.")
        return

    if verbose:
        print(f"  Collected {len(examples)} examples from {n_games} games.")

    # Cap to max_examples — a warm-start needs coverage, not exhaustive imitation.
    # The per-example forward-pass loop makes large datasets very slow on GPU.
    if max_examples is not None and len(examples) > max_examples:
        rng_sample = random.Random(seed + 1)
        examples = rng_sample.sample(examples, max_examples)
        if verbose:
            print(f"  Sampled {max_examples} examples for training.")

    optimizer = Adam(agent.model.parameters(), lr=lr)
    rng = random.Random(seed)

    agent.model.train()

    for epoch in range(n_epochs):
        rng.shuffle(examples)
        total_edge_loss = 0.0
        total_frac_loss = 0.0
        n_batches = 0

        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            edge_losses = []
            frac_losses = []

            for ex in batch:
                obs = ex.obs.to(dev)
                am = ex.acting_mask.to(dev)

                # Forward pass (no GRU state — treat each example independently)
                move_logits, alpha, beta, _, _ = agent.model(
                    obs.x, obs.edge_index, obs.edge_attr, obs.u, am,
                    h_tiles=None, turn_frac=0.5,
                )

                # Mask non-acting edges
                src = obs.edge_index[0]
                valid = am[src]
                move_logits = move_logits.masked_fill(~valid, -1e9)

                # Edge selection: cross-entropy (greedy's chosen edge = target)
                target = torch.tensor(ex.target_edge, device=dev)
                log_probs = F.log_softmax(move_logits, dim=0)
                edge_loss = F.nll_loss(log_probs.unsqueeze(0), target.unsqueeze(0))
                edge_losses.append(edge_loss)

                # Fraction: MSE between Beta mean and greedy's fraction
                ei = ex.target_edge
                beta_mean = alpha[ei] / (alpha[ei] + beta[ei])
                target_frac = torch.tensor(ex.target_frac, device=dev, dtype=torch.float32)
                frac_loss = F.mse_loss(beta_mean, target_frac)
                frac_losses.append(frac_loss)

            loss = torch.stack(edge_losses).mean() + 0.5 * torch.stack(frac_losses).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.model.parameters(), 1.0)
            optimizer.step()

            total_edge_loss += torch.stack(edge_losses).mean().item()
            total_frac_loss += torch.stack(frac_losses).mean().item()
            n_batches += 1

        if verbose:
            print(
                f"  BC epoch {epoch+1:2d}/{n_epochs}  "
                f"edge_loss={total_edge_loss/max(n_batches,1):.4f}  "
                f"frac_loss={total_frac_loss/max(n_batches,1):.4f}"
            )

    agent.model.eval()

    if verbose:
        print("BC warm-start complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BC warm-start for StrategistGNN")
    parser.add_argument("--games",   type=int, default=200)
    parser.add_argument("--epochs",  type=int, default=10)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--device",  default="cpu")
    parser.add_argument("--output",  default="runs/strategist/bc_warmstart.pt")
    args = parser.parse_args()

    model = StrategistGNN()
    agent = StrategistAgent(model=model, device=args.device)
    bc_train(agent, n_games=args.games, n_epochs=args.epochs,
             lr=args.lr, device=args.device)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    agent.save(args.output)
    print(f"Saved to {args.output}")
