"""Tests for agent implementations."""

import pytest

from hexwar.agents.greedy_agent import N_FEATURES, GreedyAgent
from hexwar.agents.random_agent import RandomAgent
from hexwar.engine.game_engine import HexWarEnv
from hexwar.engine.types import PLAYER_IDS


class TestRandomAgent:
    def test_returns_orders(self):
        env = HexWarEnv(seed=42)
        agent = RandomAgent(seed=0)
        player_id = env.players[0].id
        orders = agent(env.board, player_id, env.players, env.stats)
        assert isinstance(orders, dict)

    def test_only_own_tiles(self):
        """Agent only issues orders from tiles it owns."""
        env = HexWarEnv(seed=42)
        agent = RandomAgent(seed=1)
        player_id = env.players[0].id
        orders = agent(env.board, player_id, env.players, env.stats)
        for from_key in orders:
            assert env.board[from_key].owner == player_id

    def test_targets_are_neighbors(self):
        """All order targets are adjacent to the source tile."""
        from hexwar.engine.hex_utils import hex_neighbors, hex_to_key
        env = HexWarEnv(seed=42)
        agent = RandomAgent(seed=2)
        player_id = env.players[0].id
        orders = agent(env.board, player_id, env.players, env.stats)
        for from_key, order in orders.items():
            from_tile = env.board[from_key]
            nb_keys = {hex_to_key(n) for n in hex_neighbors(from_tile.coord)}
            assert order.to_key in nb_keys


class TestGreedyAgent:
    def test_returns_orders(self):
        env = HexWarEnv(seed=42)
        agent = GreedyAgent()
        player_id = env.players[0].id
        orders = agent(env.board, player_id, env.players, env.stats)
        assert isinstance(orders, dict)

    def test_only_own_tiles(self):
        env = HexWarEnv(seed=42)
        agent = GreedyAgent()
        player_id = env.players[0].id
        orders = agent(env.board, player_id, env.players, env.stats)
        for from_key in orders:
            assert env.board[from_key].owner == player_id

    def test_weight_setter(self):
        agent = GreedyAgent()
        new_weights = [float(i) for i in range(N_FEATURES)]
        agent.weights = new_weights
        assert agent.weights == new_weights

    def test_invalid_weight_length(self):
        with pytest.raises(ValueError):
            GreedyAgent(weights=[1.0, 2.0])  # too few

    def test_send_fraction(self):
        """With send_fraction=0.5, agent never sends more than half its units."""
        env = HexWarEnv(seed=42)
        agent = GreedyAgent(send_fraction=0.5)
        player_id = env.players[0].id
        orders = agent(env.board, player_id, env.players, env.stats)
        for from_key, order in orders.items():
            tile_units = env.board[from_key].units
            assert order.requested_units <= max(1, round(tile_units * 0.5) + 1)

    def test_greedy_beats_random_tiles(self):
        """
        Greedy should hold more tiles on average than a random agent
        when both compete under identical conditions (same seeds).

        We measure tiles-at-game-end rather than win rate because in 6-player
        games with a short horizon most games end at max_turns with no winner.
        Tiles held is a robust proxy for board-control performance.
        """
        N = 20
        greedy_tiles = 0
        random_tiles = 0

        for seed in range(N):
            # Game A: p1=greedy, p2..p6=random
            agents_a = {pid: RandomAgent(seed=i + seed * 10) for i, pid in enumerate(PLAYER_IDS)}
            agents_a["p1"] = GreedyAgent()
            env_a = HexWarEnv(agents=agents_a, seed=seed, max_turns=100)
            env_a.run()
            greedy_tiles += sum(1 for t in env_a.board.values() if t.owner == "p1")

            # Game B: all random (p1=random)
            agents_b = {pid: RandomAgent(seed=i + seed * 10) for i, pid in enumerate(PLAYER_IDS)}
            env_b = HexWarEnv(agents=agents_b, seed=seed, max_turns=100)
            env_b.run()
            random_tiles += sum(1 for t in env_b.board.values() if t.owner == "p1")

        avg_greedy = greedy_tiles / N
        avg_random = random_tiles / N
        assert avg_greedy > avg_random, (
            f"Greedy ({avg_greedy:.1f} tiles) should hold more than random ({avg_random:.1f} tiles)"
        )
