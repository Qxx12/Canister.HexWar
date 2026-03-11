"""
Elo rating system for multi-player HexWar agents.

Extends the standard 2-player Elo model to 6-player games using
the "all-pairs" decomposition: each game of N players generates
N*(N-1)/2 pairwise comparisons. The winner beats all non-winners.

Usage::

    elo = EloSystem(initial_rating=1200)
    elo.update("greedy", "random", winner="greedy")
    elo.update("neural", "greedy", winner="neural")
    print(elo.ratings)
"""

from __future__ import annotations

from collections import defaultdict


class EloSystem:
    """
    Elo rating tracker.

    Args:
        initial_rating: Starting Elo for new agents.
        k_factor:       Learning rate. Higher K → faster adaptation.
    """

    def __init__(self, initial_rating: float = 1200.0, k_factor: float = 32.0) -> None:
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.ratings: dict[str, float] = defaultdict(lambda: initial_rating)
        self.game_counts: dict[str, int] = defaultdict(int)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Expected score for player A against B (0..1)."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update(self, player_a: str, player_b: str, winner: str | None) -> None:
        """
        Update ratings after a 1v1 match.

        Args:
            player_a: One player's name.
            player_b: Other player's name.
            winner:   Name of the winner, or None for a draw.
        """
        ra = self.ratings[player_a]
        rb = self.ratings[player_b]

        ea = self.expected_score(ra, rb)
        eb = 1.0 - ea

        if winner == player_a:
            sa, sb = 1.0, 0.0
        elif winner == player_b:
            sa, sb = 0.0, 1.0
        else:  # draw
            sa, sb = 0.5, 0.5

        self.ratings[player_a] = ra + self.k_factor * (sa - ea)
        self.ratings[player_b] = rb + self.k_factor * (sb - eb)
        self.game_counts[player_a] += 1
        self.game_counts[player_b] += 1

    def update_multiplayer(
        self,
        player_ids: list[str],
        winner_id: str | None,
    ) -> None:
        """
        Update ratings for an N-player game using all-pairs comparison.

        The winner beats every non-winner. Non-winners are considered draws
        with each other (unless we have elimination order — extend for that).

        Args:
            player_ids: All players in the game (active + eliminated).
            winner_id:  The winner, or None for a draw/timeout.
        """
        for i, a in enumerate(player_ids):
            for b in player_ids[i + 1:]:
                if winner_id == a:
                    self.update(a, b, winner=a)
                elif winner_id == b:
                    self.update(a, b, winner=b)
                else:
                    self.update(a, b, winner=None)

    def print_rankings(self) -> None:
        """Print agents sorted by Elo rating."""
        print(f"\n{'Agent':<30} {'Elo':>8} {'Games':>8}")
        print("-" * 48)
        for name, rating in sorted(self.ratings.items(), key=lambda x: x[1], reverse=True):
            games = self.game_counts.get(name, 0)
            print(f"{name:<30} {rating:>8.1f} {games:>8}")
