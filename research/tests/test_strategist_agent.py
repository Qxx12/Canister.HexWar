"""Tests for StrategistAgent inference and GRU state management."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import torch

from hexwar.agents.neural.strategist_agent import StrategistAgent
from hexwar.agents.neural.strategist_model import StrategistGNN
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
    model = StrategistGNN(hidden_dim=32, n_layers=2, n_heads=2)
    return StrategistAgent(model=model, max_turns=100, deterministic=True, device="cpu")


class TestStrategistAgentReset:
    def test_reset_clears_gru_state(self, agent):
        agent._h_tiles = torch.zeros(10, 32)
        agent.reset()
        assert agent._h_tiles is None

    def test_reset_clears_turn_counter(self, agent):
        agent._turn = 42
        agent.reset()
        assert agent._turn == 0

    def test_reset_clears_node_keys(self, agent):
        agent._node_keys = ["a", "b"]
        agent.reset()
        assert agent._node_keys == []


class TestStrategistAgentCall:
    def test_returns_order_map(self, board, agent):
        orders = agent(board, PLAYER_IDS[0], _players(), {})
        assert isinstance(orders, dict)

    def test_only_own_tiles_in_orders(self, board, agent):
        player_id = PLAYER_IDS[0]
        orders = agent(board, player_id, _players(), {})
        for from_key in orders:
            assert board[from_key].owner == player_id

    def test_targets_are_neighbors(self, board, agent):
        from hexwar.engine.hex_utils import hex_neighbors, hex_to_key
        player_id = PLAYER_IDS[0]
        orders = agent(board, player_id, _players(), {})
        for from_key, order in orders.items():
            from_tile = board[from_key]
            nb_keys = {hex_to_key(n) for n in hex_neighbors(from_tile.coord)}
            assert order.to_key in nb_keys

    def test_units_to_send_positive(self, board, agent):
        player_id = PLAYER_IDS[0]
        orders = agent(board, player_id, _players(), {})
        for order in orders.values():
            assert order.requested_units >= 1

    def test_turn_counter_increments(self, board, agent):
        agent.reset()
        assert agent._turn == 0
        agent(board, PLAYER_IDS[0], _players(), {})
        assert agent._turn == 1
        agent(board, PLAYER_IDS[0], _players(), {})
        assert agent._turn == 2

    def test_gru_state_stored_after_call(self, board, agent):
        agent.reset()
        assert agent._h_tiles is None
        agent(board, PLAYER_IDS[0], _players(), {})
        assert agent._h_tiles is not None

    def test_gru_state_changes_between_calls(self, board, agent):
        agent.reset()
        agent(board, PLAYER_IDS[0], _players(), {})
        h1 = agent._h_tiles.clone()
        agent(board, PLAYER_IDS[0], _players(), {})
        h2 = agent._h_tiles.clone()
        # GRU state must update each turn
        assert not torch.allclose(h1, h2)


class TestStrategistAgentDeterminism:
    def test_deterministic_mode_same_output(self, board):
        import copy
        model = StrategistGNN(hidden_dim=32, n_layers=2, n_heads=2)
        model.eval()
        model_b = copy.deepcopy(model)

        a = StrategistAgent(model=model,   deterministic=True)
        b = StrategistAgent(model=model_b, deterministic=True)

        player_id = PLAYER_IDS[0]
        players = _players()
        a.reset()
        b.reset()

        orders_a = a(board, player_id, players, {})
        orders_b = b(board, player_id, players, {})

        assert set(orders_a.keys()) == set(orders_b.keys())
        for k in orders_a:
            assert orders_a[k].to_key == orders_b[k].to_key
            assert orders_a[k].requested_units == orders_b[k].requested_units

    def test_stochastic_mode_runs(self, board):
        model = StrategistGNN(hidden_dim=32, n_layers=2, n_heads=2)
        agent = StrategistAgent(model=model, deterministic=False)
        orders = agent(board, PLAYER_IDS[0], _players(), {})
        assert isinstance(orders, dict)


class TestStrategistAgentGRUAlignment:
    def test_gru_state_alignment_survives_same_board(self, board, agent):
        """GRU state is reused when node ordering is unchanged."""
        player_id = PLAYER_IDS[0]
        players = _players()
        agent.reset()
        agent(board, player_id, players, {})
        keys_before = list(agent._node_keys)
        agent(board, player_id, players, {})
        assert agent._node_keys == keys_before

    def test_gru_state_rebuilt_on_key_change(self, agent):
        """New node keys trigger GRU state rebuild with zeros for unknown tiles."""
        agent._h_tiles = torch.ones(3, 32)
        agent._node_keys = ["a", "b", "c"]

        new_keys = ["a", "b", "c", "d"]   # one new tile
        h = agent._get_aligned_h(new_keys, 4, torch.device("cpu"))
        assert h.shape == (4, 32)
        # Existing tiles copied
        assert torch.allclose(h[0], torch.ones(32))
        assert torch.allclose(h[1], torch.ones(32))
        assert torch.allclose(h[2], torch.ones(32))
        # New tile starts at zero
        assert torch.allclose(h[3], torch.zeros(32))


class TestStrategistAgentPersistence:
    def test_save_load_roundtrip(self, tmp_path, board):
        model = StrategistGNN(hidden_dim=32, n_layers=2, n_heads=2)
        agent = StrategistAgent(model=model, deterministic=True)

        path = tmp_path / "strategist.pt"
        agent.save(path)

        import copy
        agent2 = StrategistAgent(
            model=copy.deepcopy(model), deterministic=True
        )
        agent2.load(path)

        # Both agents should produce identical outputs after loading same weights
        player_id = PLAYER_IDS[0]
        players = _players()
        agent.reset()
        agent2.reset()
        o1 = agent(board, player_id, players, {})
        o2 = agent2(board, player_id, players, {})
        assert set(o1.keys()) == set(o2.keys())
