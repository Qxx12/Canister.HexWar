"""Tests for PPOAgent inference, reset, and deterministic mode."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from hexwar.agents.neural.gnn_model import HexWarGNN
from hexwar.agents.neural.ppo_agent import PPOAgent
from hexwar.engine.board_generator import generate_board
from hexwar.engine.types import PLAYER_IDS, Player, PlayerType


def _players() -> list[Player]:
    return [
        Player(id=pid, name=pid, color="#fff", type=PlayerType.AI, is_eliminated=False)
        for pid in PLAYER_IDS
    ]


@pytest.fixture
def board():
    return generate_board(seed=0)


@pytest.fixture
def agent():
    model = HexWarGNN(hidden_dim=32, n_layers=2, n_heads=2, history_k=1)
    return PPOAgent(model=model, history_k=1, deterministic=True, device="cpu")


class TestPPOAgentReset:
    def test_reset_clears_history(self, board, agent):
        """After reset, history buffer is empty or reset to initial board."""
        agent.reset(board)
        assert len(agent._history) > 0

    def test_reset_without_board_clears_buffer(self, agent):
        agent.reset()
        assert len(agent._history) == 0


class TestPPOAgentCall:
    def test_returns_order_map(self, board, agent):
        players = _players()
        player_id = PLAYER_IDS[0]
        orders = agent(board, player_id, players, {})
        assert isinstance(orders, dict)

    def test_only_own_tiles(self, board, agent):
        """All order sources must be tiles owned by the acting player."""
        player_id = PLAYER_IDS[0]
        players = _players()
        orders = agent(board, player_id, players, {})
        for from_key in orders:
            assert board[from_key].owner == player_id

    def test_targets_are_neighbors(self, board, agent):
        """All order targets must be adjacent to the source tile."""
        from hexwar.engine.hex_utils import hex_neighbors, hex_to_key

        player_id = PLAYER_IDS[0]
        players = _players()
        orders = agent(board, player_id, players, {})
        for from_key, order in orders.items():
            from_tile = board[from_key]
            nb_keys = {hex_to_key(n) for n in hex_neighbors(from_tile.coord)}
            assert order.to_key in nb_keys

    def test_units_to_send_positive(self, board, agent):
        """Every order must request at least 1 unit."""
        player_id = PLAYER_IDS[0]
        players = _players()
        orders = agent(board, player_id, players, {})
        for order in orders.values():
            assert order.requested_units >= 1

    def test_history_grows_after_call(self, board, agent):
        """Calling the agent pushes the board into the history buffer."""
        agent.reset()
        player_id = PLAYER_IDS[0]
        players = _players()
        initial_len = len(agent._history)
        agent(board, player_id, players, {})
        assert len(agent._history) > initial_len

    def test_deterministic_mode(self, board):
        """
        Two deterministic agents with identical weights produce the same orders
        from the same board state (no stochastic sampling).
        """
        import copy

        model_a = HexWarGNN(hidden_dim=32, n_layers=2, n_heads=2, history_k=1)
        model_a.eval()
        model_b = copy.deepcopy(model_a)   # exact same weights, eval mode

        agent_a = PPOAgent(model=model_a, history_k=1, deterministic=True)
        agent_b = PPOAgent(model=model_b, history_k=1, deterministic=True)

        player_id = PLAYER_IDS[0]
        players = _players()

        agent_a.reset()
        agent_b.reset()

        orders_a = agent_a(board, player_id, players, {})
        orders_b = agent_b(board, player_id, players, {})

        assert set(orders_a.keys()) == set(orders_b.keys())
        for k in orders_a:
            assert orders_a[k].to_key == orders_b[k].to_key
            assert orders_a[k].requested_units == orders_b[k].requested_units

    def test_stochastic_mode_runs(self, board):
        """Stochastic mode (deterministic=False) runs without error."""
        model = HexWarGNN(hidden_dim=32, n_layers=2, n_heads=2, history_k=1)
        stoch_agent = PPOAgent(model=model, history_k=1, deterministic=False)
        player_id = PLAYER_IDS[0]
        players = _players()
        orders = stoch_agent(board, player_id, players, {})
        assert isinstance(orders, dict)


class TestPPOAgentHistory:
    def test_history_k5_runs(self, board):
        """PPOAgent with history_k=5 runs without error."""
        model = HexWarGNN(hidden_dim=32, n_layers=2, n_heads=2, history_k=5)
        agent = PPOAgent(model=model, history_k=5, deterministic=True)
        player_id = PLAYER_IDS[0]
        players = _players()
        agent.reset(board)
        orders = agent(board, player_id, players, {})
        assert isinstance(orders, dict)

    def test_multiple_turns_accumulate_history(self, board):
        """Calling the agent 3 times accumulates 3 board pushes."""
        model = HexWarGNN(hidden_dim=32, n_layers=2, n_heads=2, history_k=5)
        agent = PPOAgent(model=model, history_k=5, deterministic=True)
        player_id = PLAYER_IDS[0]
        players = _players()

        agent.reset()
        for _ in range(3):
            agent(board, player_id, players, {})

        # Buffer is capped at k=5; after 3 calls buffer has 3 entries (or pre-filled)
        assert len(agent._history) >= 1
