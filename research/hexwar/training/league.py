"""
League — diverse opponent pool for StrategistAgent training.

A League maintains a heterogeneous pool of opponent agents:
  - Historical snapshots of the StrategistAgent (past checkpoints)
  - Evolved GreedyAgent instances (CMA-ES best weights)
  - The current learning policy itself (self-play)

At the start of each episode, opponents are sampled from the pool according
to the configured mixing probabilities, preventing the learner from exploiting
a fixed opponent and encouraging robustness across diverse play styles.

Mixing schedule (default):
  - 50% — current learner (self-play)
  - 30% — random historical snapshot
  - 20% — greedy/evolved agent from the greedy pool

The mixing ratios shift over training: as the learner improves, historical
snapshot weight increases so the agent does not forget how to beat weaker
play.

Usage::

    league = League(learner)
    league.add_greedy(GreedyAgent(weights=best_weights))
    league.snapshot()   # save current policy as a checkpoint

    # Inside training loop:
    episode_agents = league.sample_opponents(learner_id="p1")
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

from ..agents.base_agent import BaseAgent


@dataclass
class LeagueConfig:
    self_play_prob: float = 0.50    # fraction of opponent slots filled by learner copy
    snapshot_prob: float = 0.30     # fraction filled by random historical snapshot
    greedy_prob: float = 0.20       # fraction filled by greedy/evolved agents

    max_snapshots: int = 20         # maximum number of past checkpoints to keep
    snapshot_interval: int = 100    # training updates between auto-snapshots


class League:
    """
    Manages a heterogeneous opponent pool for league training.

    Args:
        learner:     The agent being trained.
        config:      Mixing and pool size configuration.
        seed:        RNG seed for reproducibility.
    """

    def __init__(
        self,
        learner: BaseAgent,
        config: LeagueConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.learner = learner
        self.config = config or LeagueConfig()
        self._rng = random.Random(seed)
        self._snapshots: list[BaseAgent] = []
        self._greedy_pool: list[BaseAgent] = []
        self._update_count: int = 0

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    def snapshot(self) -> None:
        """
        Save a deep-copy of the current learner as a past-checkpoint opponent.
        Caps pool at max_snapshots by evicting the oldest entry.
        """
        snap = copy.deepcopy(self.learner)
        # Put into eval mode if the model supports it
        if hasattr(snap, "model"):
            snap.model.eval()  # type: ignore[attr-defined]
        self._snapshots.append(snap)
        if len(self._snapshots) > self.config.max_snapshots:
            self._snapshots.pop(0)

    def add_greedy(self, agent: BaseAgent) -> None:
        """Add a greedy / evolved agent to the pool."""
        self._greedy_pool.append(agent)

    def on_update(self) -> None:
        """Call after each training update step for auto-snapshotting."""
        self._update_count += 1
        if (
            self.config.snapshot_interval > 0
            and self._update_count % self.config.snapshot_interval == 0
        ):
            self.snapshot()

    # ------------------------------------------------------------------
    # Episode opponent sampling
    # ------------------------------------------------------------------

    def sample_opponents(
        self,
        learner_id: str,
        all_player_ids: list[str],
    ) -> dict[str, BaseAgent]:
        """
        Return a mapping of player_id → agent for one episode.

        The learner occupies learner_id. All other slots are filled by
        sampling from the pool according to the configured mixing probabilities.

        Args:
            learner_id:     The player slot reserved for the learning agent.
            all_player_ids: All player IDs in the game.

        Returns:
            Dict[player_id, agent] for all players.
        """
        agents: dict[str, BaseAgent] = {}
        for pid in all_player_ids:
            if pid == learner_id:
                agents[pid] = self.learner
            else:
                agents[pid] = self._sample_one()
        return agents

    def _sample_one(self) -> BaseAgent:
        """Sample a single opponent from the pool."""
        weights: list[float] = []
        pools: list[list[BaseAgent]] = []

        # Self-play: always available
        weights.append(self.config.self_play_prob)
        pools.append([self.learner])

        # Snapshots: only available if we have some
        if self._snapshots:
            weights.append(self.config.snapshot_prob)
            pools.append(self._snapshots)
        else:
            # Shift weight to self-play if no snapshots yet
            weights[0] += self.config.snapshot_prob

        # Greedy pool: only if populated
        if self._greedy_pool:
            weights.append(self.config.greedy_prob)
            pools.append(self._greedy_pool)
        else:
            weights[0] += self.config.greedy_prob

        # Weighted-random pool selection, then uniform within that pool
        chosen_pool = self._rng.choices(pools, weights=weights, k=1)[0]
        return self._rng.choice(chosen_pool)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        return {
            "n_snapshots": len(self._snapshots),
            "n_greedy": len(self._greedy_pool),
            "updates": self._update_count,
        }
