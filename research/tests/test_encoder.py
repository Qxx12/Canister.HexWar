"""Tests for the board encoder and history buffer."""

import pytest

from hexwar.agents.neural.encoder import (
    N_EDGE_FEATURES,
    N_GLOBAL_FEATURES,
    N_NODE_FEATURES,
    encode_board,
    encode_board_with_history,
    node_feature_dim,
)
from hexwar.agents.neural.history_buffer import N_HIST_FEATURES, HistoryBuffer
from hexwar.engine.board_generator import generate_board


@pytest.fixture
def board():
    return generate_board(seed=42)


class TestNodeFeatureDim:
    def test_k1_is_base(self):
        assert node_feature_dim(1) == N_NODE_FEATURES

    def test_k5(self):
        assert node_feature_dim(5) == N_NODE_FEATURES + 4 * N_HIST_FEATURES

    def test_k3(self):
        assert node_feature_dim(3) == N_NODE_FEATURES + 2 * N_HIST_FEATURES


class TestHistoryBuffer:
    def test_initial_fill(self, board):
        buf = HistoryBuffer(k=5)
        buf.reset(board)
        frames = buf.get_frames()
        assert len(frames) == 5

    def test_push_advances(self, board):
        buf = HistoryBuffer(k=3)
        buf.reset(board)
        b2 = generate_board(seed=1)
        buf.push(b2)
        frames = buf.get_frames()
        assert len(frames) == 3
        # Most recent frame should be b2
        assert frames[-1] is not frames[-2]

    def test_oldest_dropped(self, board):
        buf = HistoryBuffer(k=2)
        buf.reset(board)
        b2 = generate_board(seed=1)
        b3 = generate_board(seed=2)
        buf.push(b2)
        buf.push(b3)
        frames = buf.get_frames()
        # Only k=2 frames retained
        assert len(frames) == 2

    def test_snapshot_independence(self, board):
        """Mutating the board after push should not affect stored snapshot."""
        buf = HistoryBuffer(k=2)
        buf.reset(board)
        buf.push(board)
        key = next(iter(board))
        original_units = buf.get_frames()[-1][key].units
        board[key].units += 999
        assert buf.get_frames()[-1][key].units == original_units


class TestEncodeBoard:
    def test_single_frame_shape(self, board):
        pytest.importorskip("torch_geometric")
        data = encode_board(board, "p1")
        n = len(board)
        assert data.x.shape == (n, N_NODE_FEATURES)
        assert data.edge_index.shape[0] == 2
        assert data.edge_attr.shape[1] == N_EDGE_FEATURES
        assert data.u.shape == (N_GLOBAL_FEATURES,)

    def test_node_keys_match_board(self, board):
        pytest.importorskip("torch_geometric")
        data = encode_board(board, "p1")
        assert set(data.node_keys) == set(board.keys())

    def test_features_in_range(self, board):
        pytest.importorskip("torch_geometric")
        data = encode_board(board, "p1")
        # All features should be finite
        assert data.x.isfinite().all()
        assert data.edge_attr.isfinite().all()


class TestEncodeWithHistory:
    def test_stacked_feature_dim(self, board):
        pytest.importorskip("torch_geometric")
        k = 5
        frames = [board] * k
        data = encode_board_with_history(frames, "p1")
        n = len(board)
        assert data.x.shape == (n, node_feature_dim(k))

    def test_k1_equals_single_frame(self, board):
        pytest.importorskip("torch_geometric")
        import torch
        d1 = encode_board(board, "p1")
        d2 = encode_board_with_history([board], "p1")
        assert torch.allclose(d1.x, d2.x)

    def test_history_differs_from_current(self, board):
        """Historical frames with different unit counts produce different features."""
        pytest.importorskip("torch_geometric")
        from copy import deepcopy

        import torch

        old_board = deepcopy(board)
        # Bump all units by 5 in the "current" frame
        for tile in board.values():
            if tile.owner is not None:
                tile.units += 5

        data = encode_board_with_history([old_board, old_board, board], "p1")
        # Historical columns (offset 18 onward) should differ from current
        current_units = data.x[:, 0]   # col 0 = current units
        hist_units    = data.x[:, N_NODE_FEATURES]  # col 18 = t-1 units (most recent hist)
        # At least some tiles should differ
        assert not torch.allclose(current_units, hist_units)

    def test_all_features_finite(self, board):
        pytest.importorskip("torch_geometric")
        frames = [board] * 5
        data = encode_board_with_history(frames, "p1")
        assert data.x.isfinite().all()
        assert data.edge_attr.isfinite().all()
        assert data.u.isfinite().all()
