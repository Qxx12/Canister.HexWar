"""Tests for agent implementations."""

import pytest

from hexwar.agents.evolutionary.fitness import _build_agents
from hexwar.agents.greedy_agent import DEFAULT_WEIGHTS, N_FEATURES, GreedyAgent
from hexwar.agents.random_agent import RandomAgent
from hexwar.engine.game_engine import HexWarEnv
from hexwar.engine.types import PLAYER_IDS, AxialCoord, Tile


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


# ---------------------------------------------------------------------------
# Helpers for hand-crafted board tests
# ---------------------------------------------------------------------------

def _tile(q: int, r: int, owner: str | None, units: int,
          is_start: bool = False) -> tuple[str, Tile]:
    key = f"{q},{r}"
    t = Tile(
        coord=AxialCoord(q, r),
        owner=owner,
        units=units,
        is_start_tile=is_start,
        start_owner=owner if is_start else None,
        terrain="plains",
        newly_conquered=False,
    )
    return key, t


def _board(*tiles: tuple[str, Tile]) -> dict:
    return dict(tiles)


def _weights_only(feat: int, value: float = 10.0) -> list[float]:
    """Return a weight vector with only feature `feat` set (all others 0)."""
    w = [0.0] * N_FEATURES
    w[feat] = value
    return w


# ---------------------------------------------------------------------------
# Tests for v4 features (8–10) and BFS gradient
# ---------------------------------------------------------------------------

class TestStartGradient:
    def test_returns_inv_dist(self):
        """BFS gradient values equal 1/(1+dist) from the unowned start tile."""
        agent = GreedyAgent()
        board = _board(
            _tile(0, 0, "p1", 5),           # source — dist=2 from start
            _tile(1, 0, None, 0),            # neutral — dist=1 from start
            _tile(2, 0, None, 0, is_start=True),  # unowned start — dist=0
        )
        grad = agent._compute_start_gradient(board, "p1")
        assert abs(grad["2,0"] - 1.0)  < 1e-9   # dist=0 → 1.0
        assert abs(grad["1,0"] - 0.5)  < 1e-9   # dist=1 → 0.5
        assert abs(grad["0,0"] - 1/3)  < 1e-9   # dist=2 → 0.333…

    def test_empty_when_all_starts_owned(self):
        """Gradient is empty when the player owns every start tile."""
        agent = GreedyAgent()
        board = _board(
            _tile(0, 0, "p1", 5, is_start=True),
            _tile(1, 0, None, 0),
        )
        assert agent._compute_start_gradient(board, "p1") == {}

    def test_multi_source_uses_nearest(self):
        """Two unowned start tiles — each tile gets distance to the nearest."""
        # hex neighbours: (q+1,r),(q-1,r),(q,r+1),(q,r-1),(q+1,r-1),(q-1,r+1)
        # tile at (0,0): dist=1 from start at (-1,0) and dist=1 from (1,0)
        agent = GreedyAgent()
        board = _board(
            _tile(-1, 0, "p2", 1, is_start=True),  # unowned start A
            _tile( 1, 0, "p3", 1, is_start=True),  # unowned start B
            _tile( 0, 0, "p1", 5),                 # between both — dist=1 from each
        )
        grad = agent._compute_start_gradient(board, "p1")
        assert abs(grad["0,0"] - 0.5) < 1e-9   # nearest dist = 1


class TestInvDistFeature:
    def test_prefers_tile_adjacent_to_unowned_start(self):
        """
        Feature 8 alone should pull the agent toward the tile adjacent to
        the unowned start tile rather than an equally accessible neutral.

        Board (hex neighbours of (0,0): (1,0),(−1,0),(0,1),(0,−1),(1,−1),(−1,1)):
          (0,0)  p1  10 units   — source
          (1,0)  neutral        — 1 hop from unowned start at (2,0)  → inv_dist=0.5
          (0,-1) neutral        — not adjacent to any start tile     → inv_dist=0.25
          (2,0)  unowned start
        """
        # Only feature 8 active, strongly positive
        w = _weights_only(8, 10.0)
        agent = GreedyAgent(weights=w)
        from hexwar.engine.types import Player, PlayerType
        board = _board(
            _tile(0,  0, "p1", 10),
            _tile(1,  0, None,  0),
            _tile(0, -1, None,  0),
            _tile(2,  0, None,  0, is_start=True),
        )
        players = [Player(id="p1", name="P1", color="#f00",
                          type=PlayerType.AI, is_eliminated=False)]
        orders = agent(board, "p1", players, {})
        assert orders["0,0"].to_key == "1,0"


class TestNearElimFeature:
    def test_prefers_near_elim_enemy(self):
        """
        Feature 9: near-elim bonus should make the agent prefer an enemy with
        ≤ 4 tiles over an equally strong enemy with many tiles.
        """
        # Two enemies, both beatable (p1 has 10 units, enemies have 5 each).
        # p0 has only 2 tiles → near-elim; p2 has 10 tiles → not.
        w = _weights_only(9, 10.0)   # only near-elim feature active
        agent = GreedyAgent(weights=w)
        from hexwar.engine.types import Player, PlayerType

        p2_extras = [_tile(100 + i, 0, "p2", 1) for i in range(9)]
        board = _board(
            _tile(0,  0, "p1", 10),
            _tile(1,  0, "p0",  5),   # near-elim enemy (total 2 tiles)
            _tile(0, -1, "p2",  5),   # healthy enemy (total 10 tiles)
            _tile(5,  5, "p0",  1),   # p0's second tile
            *p2_extras,
        )
        players = [
            Player(id="p1", name="P1", color="#f00", type=PlayerType.AI, is_eliminated=False),
            Player(id="p0", name="P0", color="#0f0", type=PlayerType.AI, is_eliminated=False),
            Player(id="p2", name="P2", color="#00f", type=PlayerType.AI, is_eliminated=False),
        ]
        orders = agent(board, "p1", players, {})
        assert orders["0,0"].to_key == "1,0"   # attacks near-elim p0

    def test_threshold_boundary(self):
        """
        Enemy at exactly NEAR_ELIM_THRESHOLD tiles gets the bonus;
        enemy at NEAR_ELIM_THRESHOLD+1 does not.
        Both are equally conquerable — only the bonus distinguishes them.
        """
        from hexwar.agents.greedy_agent import NEAR_ELIM_THRESHOLD
        from hexwar.engine.types import Player, PlayerType

        w = _weights_only(9, 10.0)
        agent = GreedyAgent(weights=w)

        # p0 has exactly NEAR_ELIM_THRESHOLD tiles (including the one at (1,0))
        p0_extra = [_tile(100 + i, 0, "p0", 1) for i in range(NEAR_ELIM_THRESHOLD - 1)]
        # p2 has NEAR_ELIM_THRESHOLD+1 tiles (one more → no bonus)
        p2_extra = [_tile(200 + i, 0, "p2", 1) for i in range(NEAR_ELIM_THRESHOLD)]

        board = _board(
            _tile(0,  0, "p1", 10),
            _tile(1,  0, "p0",  5),   # at-threshold → near-elim fires
            _tile(0, -1, "p2",  5),   # above threshold → no bonus
            *p0_extra,
            *p2_extra,
        )
        players = [
            Player(id="p1", name="P1", color="#f00", type=PlayerType.AI, is_eliminated=False),
            Player(id="p0", name="P0", color="#0f0", type=PlayerType.AI, is_eliminated=False),
            Player(id="p2", name="P2", color="#00f", type=PlayerType.AI, is_eliminated=False),
        ]
        orders = agent(board, "p1", players, {})
        assert orders["0,0"].to_key == "1,0"   # prefers at-threshold p0 over above-threshold p2


class TestNeutralAdjFeature:
    def test_prefers_junction_neutral(self):
        """
        Feature 10: agent prefers a neutral tile surrounded by more open
        neutrals (higher expansion value) over a dead-end neutral.

        (0,0) p1 source.
        (1,0) neutral — neighbours: (2,0) neutral, (1,1) neutral, (1,-1) neutral → 3 neutral adj
        (0,-1) neutral — no other neutrals adjacent → 0 neutral adj
        """
        w = _weights_only(10, 10.0)
        agent = GreedyAgent(weights=w)
        from hexwar.engine.types import Player, PlayerType

        board = _board(
            _tile(0,  0, "p1", 5),
            _tile(1,  0, None, 0),   # junction: 3 neutral neighbours below
            _tile(2,  0, None, 0),
            _tile(1,  1, None, 0),
            _tile(1, -1, None, 0),
            _tile(0, -1, None, 0),   # dead-end: neighbours are p1 and (0,-2) absent
        )
        players = [Player(id="p1", name="P1", color="#f00",
                          type=PlayerType.AI, is_eliminated=False)]
        orders = agent(board, "p1", players, {})
        assert orders["0,0"].to_key == "1,0"


# ---------------------------------------------------------------------------
# Tests for fitness._build_agents (12-dim split + opponent pool)
# ---------------------------------------------------------------------------

class TestBuildAgents:
    def test_splits_send_fraction_from_weights(self):
        """
        _build_agents must split a 12-dim vector: [:11] → feature weights,
        [11] → send_fraction (clamped to [0.5, 1.0]).
        """
        weights_12d = list(DEFAULT_WEIGHTS) + [0.75]   # send_fraction = 0.75
        agents = _build_agents(weights_12d, candidate_id="p1")
        candidate = agents["p1"]
        assert len(candidate.weights) == N_FEATURES
        assert abs(candidate._send_fraction - 0.75) < 1e-9

    def test_send_fraction_clamped_low(self):
        """Values below 0.5 are clamped to 0.5."""
        weights_12d = list(DEFAULT_WEIGHTS) + [0.1]
        agents = _build_agents(weights_12d, candidate_id="p1")
        assert agents["p1"]._send_fraction == 0.5

    def test_send_fraction_clamped_high(self):
        """Values above 1.0 are clamped to 1.0."""
        weights_12d = list(DEFAULT_WEIGHTS) + [1.5]
        agents = _build_agents(weights_12d, candidate_id="p1")
        assert agents["p1"]._send_fraction == 1.0

    def test_opponents_use_default_weights_when_none(self):
        """Without opponent_weights, all 5 opponents use DEFAULT_WEIGHTS."""
        weights_12d = list(DEFAULT_WEIGHTS) + [1.0]
        agents = _build_agents(weights_12d, candidate_id="p1")
        for pid in PLAYER_IDS:
            if pid != "p1":
                assert agents[pid].weights == list(DEFAULT_WEIGHTS)

    def test_opponent_pool_mixing(self):
        """opponent_weights list is applied to non-candidate slots in order."""
        evolved = [float(i) for i in range(N_FEATURES)]   # 11-dim
        opp_pool = [list(evolved)] * 3 + [list(DEFAULT_WEIGHTS)] * 2
        weights_12d = list(DEFAULT_WEIGHTS) + [1.0]
        agents = _build_agents(weights_12d, candidate_id="p1",
                               opponent_weights=opp_pool)
        opp_weights = [agents[pid].weights for pid in PLAYER_IDS if pid != "p1"]
        assert opp_weights[:3] == [evolved] * 3
        assert opp_weights[3:] == [list(DEFAULT_WEIGHTS)] * 2

    def test_short_opponent_weights_zero_padded(self):
        """8-dim checkpoint weights are zero-padded to N_FEATURES (11)."""
        old_8d = [1.0] * 8
        opp_pool = [old_8d] * 5
        weights_12d = list(DEFAULT_WEIGHTS) + [1.0]
        agents = _build_agents(weights_12d, candidate_id="p1",
                               opponent_weights=opp_pool)
        for pid in PLAYER_IDS:
            if pid != "p1":
                w = agents[pid].weights
                assert len(w) == N_FEATURES
                assert w[:8] == [1.0] * 8
                assert w[8:] == [0.0, 0.0, 0.0]
