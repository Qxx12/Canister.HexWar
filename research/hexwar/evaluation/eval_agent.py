"""
Evaluation: pit a trained PPOAgent against GreedyAgent baseline.

Runs a round-robin tournament across all 6 player slots and reports
win rate, average tiles held, and average turns per game.

Usage::

    python -m hexwar.evaluation.eval_agent --checkpoint runs/ppo/ckpt_iter0100.pt
    python -m hexwar.evaluation.eval_agent --checkpoint runs/ppo/ckpt_iter0100.pt --games 200
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from ..agents.greedy_agent import DEFAULT_WEIGHTS, GreedyAgent
from ..agents.neural.gnn_model import HexWarGNN
from ..agents.neural.ppo_agent import PPOAgent
from ..engine.game_engine import HexWarEnv
from ..engine.types import PLAYER_IDS


@dataclass
class EvalResult:
    win_rate: float
    avg_tiles: float
    avg_turns: float
    n_games: int

    def __str__(self) -> str:
        return (
            f"Games: {self.n_games}  |  "
            f"Win%: {self.win_rate * 100:.1f}  |  "
            f"Avg tiles: {self.avg_tiles:.1f}  |  "
            f"Avg turns: {self.avg_turns:.1f}"
        )


def evaluate(
    agent: PPOAgent,
    n_games: int = 200,
    max_turns: int = 300,
    seed_offset: int = 0,
    verbose: bool = True,
) -> EvalResult:
    """
    Evaluate a PPOAgent against 5 default-weight GreedyAgent opponents.

    The agent rotates through all 6 player slots evenly to remove
    positional bias (same approach as CMA-ES fitness evaluation).

    Args:
        agent:        The trained PPOAgent to evaluate.
        n_games:      Total number of games to play.
        max_turns:    Hard turn cap per game.
        seed_offset:  Base RNG seed.
        verbose:      Print per-game progress.

    Returns:
        EvalResult with aggregated statistics.
    """
    wins = tiles_total = turns_total = 0

    for game_idx in range(n_games):
        candidate_slot = game_idx % len(PLAYER_IDS)
        candidate_id = PLAYER_IDS[candidate_slot]

        agents = {candidate_id: agent}
        for pid in PLAYER_IDS:
            if pid != candidate_id:
                agents[pid] = GreedyAgent(weights=DEFAULT_WEIGHTS)

        # Reset agent history buffer between games
        agent.reset()

        env = HexWarEnv(agents=agents, seed=seed_offset + game_idx, max_turns=max_turns)
        result = env.run()

        won = result.winner_id == candidate_id
        tiles = sum(1 for t in env.board.values() if t.owner == candidate_id)

        wins += int(won)
        tiles_total += tiles
        turns_total += result.turns_played

        if verbose and (game_idx + 1) % 20 == 0:
            running_wr = wins / (game_idx + 1)
            print(
                f"  game {game_idx + 1:4d}/{n_games}  "
                f"running win%={running_wr * 100:.1f}",
                flush=True,
            )

    return EvalResult(
        win_rate=wins / n_games,
        avg_tiles=tiles_total / n_games,
        avg_turns=turns_total / n_games,
        n_games=n_games,
    )


def evaluate_vs_greedy_baseline(n_games: int = 200, max_turns: int = 300) -> EvalResult:
    """
    Evaluate the DEFAULT_WEIGHTS GreedyAgent against itself as a sanity-check
    baseline. Expected win rate ≈ 1/6 ≈ 16.7% in a fair 6-player game.
    """
    from ..agents.greedy_agent import GreedyAgent as _G

    class _GreedyWrapper:
        """Thin wrapper to match PPOAgent interface (reset() + __call__)."""
        def __init__(self) -> None:
            self._agent = _G(weights=DEFAULT_WEIGHTS)

        def reset(self) -> None:
            pass

        def __call__(self, board, player_id, players, stats):
            return self._agent(board, player_id, players, stats)

    # Avoid circular usage by running a direct tournament
    wins = tiles_total = turns_total = 0
    for game_idx in range(n_games):
        candidate_slot = game_idx % len(PLAYER_IDS)
        candidate_id = PLAYER_IDS[candidate_slot]
        agents = {pid: _G(weights=DEFAULT_WEIGHTS) for pid in PLAYER_IDS}
        env = HexWarEnv(agents=agents, seed=game_idx, max_turns=max_turns)
        result = env.run()
        won = result.winner_id == candidate_id
        tiles = sum(1 for t in env.board.values() if t.owner == candidate_id)
        wins += int(won)
        tiles_total += tiles
        turns_total += result.turns_played

    return EvalResult(
        win_rate=wins / n_games,
        avg_tiles=tiles_total / n_games,
        avg_turns=turns_total / n_games,
        n_games=n_games,
    )


if __name__ == "__main__":
    import torch

    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent vs GreedyAgent")
    parser.add_argument("--checkpoint",  required=True, help="Path to .pt model checkpoint")
    parser.add_argument("--games",       type=int, default=200)
    parser.add_argument("--max-turns",   type=int, default=300)
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--baseline",    action="store_true",
                        help="Also run greedy baseline for comparison")
    args = parser.parse_args()

    dev = torch.device(args.device)

    model = HexWarGNN()
    ppo = PPOAgent(model=model, deterministic=True, device=dev)
    ppo.load(args.checkpoint)
    ppo.model.eval()

    print(f"Evaluating {args.checkpoint} over {args.games} games...", flush=True)
    result = evaluate(ppo, n_games=args.games, max_turns=args.max_turns)
    print(f"\nPPO agent:    {result}")

    if args.baseline:
        print("\nRunning greedy baseline...", flush=True)
        base = evaluate_vs_greedy_baseline(n_games=args.games, max_turns=args.max_turns)
        print(f"Greedy baseline: {base}")
        print(f"\nImprovement: +{(result.win_rate - base.win_rate) * 100:.1f}pp win rate")
