"""
Round-robin tournament evaluation.

Runs a full round-robin between a set of agents and reports win rates,
mean game lengths, and tiles held at end.

Each "game" is a 6-player HexWar game. For multi-agent comparisons, each
agent is assigned to all 6 slots in rotation across multiple games so that
no agent benefits from a fixed starting position.

Usage::

    from hexwar.agents.random_agent import RandomAgent
    from hexwar.agents.greedy_agent import GreedyAgent
    from hexwar.evaluation.tournament import run_tournament

    agents = {"random": RandomAgent(), "greedy": GreedyAgent()}
    results = run_tournament(agents, n_games=50)
    print(results)
"""

from __future__ import annotations

from dataclasses import dataclass

from ..agents.base_agent import BaseAgent
from ..engine.game_engine import HexWarEnv
from ..engine.types import PLAYER_IDS


@dataclass
class AgentStats:
    name: str
    wins: int = 0
    games: int = 0
    total_turns: int = 0
    total_tiles: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def avg_turns(self) -> float:
        return self.total_turns / self.games if self.games else 0.0

    @property
    def avg_tiles(self) -> float:
        return self.total_tiles / self.games if self.games else 0.0


def run_tournament(
    agents: dict[str, BaseAgent],
    n_games: int = 100,
    max_turns: int = 300,
    seed: int = 0,
) -> dict[str, AgentStats]:
    """
    Evaluate a set of agents in a round-robin tournament.

    Each game has all 6 player slots filled. If fewer than 6 agents are
    provided, agents are cycled to fill all slots.

    Args:
        agents:    Dict of name → BaseAgent.
        n_games:   Total number of games to run.
        max_turns: Hard turn cap per game.
        seed:      Base RNG seed.

    Returns:
        Dict of name → AgentStats with win rates and averages.
    """
    names = list(agents.keys())
    stats = {name: AgentStats(name=name) for name in names}

    for game_idx in range(n_games):
        # Assign agents to player slots (cycle if < 6 agents)
        slot_names = [names[i % len(names)] for i in range(len(PLAYER_IDS))]
        # Rotate starting assignment per game to reduce positional bias
        offset = game_idx % len(names)
        slot_names = slot_names[offset:] + slot_names[:offset]

        game_agents = {
            pid: agents[slot_names[i]]
            for i, pid in enumerate(PLAYER_IDS)
        }
        pid_to_name = {
            pid: slot_names[i]
            for i, pid in enumerate(PLAYER_IDS)
        }

        env = HexWarEnv(
            agents=game_agents,
            seed=seed + game_idx,
            max_turns=max_turns,
        )
        result = env.run()

        for pid, name in pid_to_name.items():
            s = stats[name]
            s.games += 1
            s.total_turns += result.turns_played
            s.total_tiles += sum(1 for t in env.board.values() if t.owner == pid)
            if result.winner_id == pid:
                s.wins += 1

    return stats


def print_tournament_results(stats: dict[str, AgentStats]) -> None:
    """Pretty-print tournament results sorted by win rate."""
    print(f"\n{'Agent':<20} {'Win%':>6} {'Wins':>6} {'Games':>6} {'AvgTurns':>10} {'AvgTiles':>10}")
    print("-" * 62)
    for s in sorted(stats.values(), key=lambda s: s.win_rate, reverse=True):
        print(
            f"{s.name:<20} {s.win_rate*100:>5.1f}% {s.wins:>6} {s.games:>6} "
            f"{s.avg_turns:>10.1f} {s.avg_tiles:>10.1f}"
        )
