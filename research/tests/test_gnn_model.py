"""Tests for HexWarGNN model architecture and forward pass."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import torch

from hexwar.agents.neural.encoder import N_EDGE_FEATURES, N_GLOBAL_FEATURES, node_feature_dim
from hexwar.agents.neural.gnn_model import HexWarGNN


def _make_batch(n_nodes: int = 8, n_edges: int = 12, history_k: int = 1):
    """Build minimal synthetic graph tensors for a forward pass."""
    node_dim = node_feature_dim(history_k)
    x = torch.randn(n_nodes, node_dim)
    # Random directed edges (staying in bounds)
    src = torch.randint(0, n_nodes, (n_edges,))
    dst = torch.randint(0, n_nodes, (n_edges,))
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = torch.randn(n_edges, N_EDGE_FEATURES)
    u = torch.randn(N_GLOBAL_FEATURES)
    acting_mask = torch.zeros(n_nodes, dtype=torch.bool)
    acting_mask[:3] = True   # first 3 nodes are acting player's
    return x, edge_index, edge_attr, u, acting_mask


class TestHexWarGNNInit:
    def test_default_instantiation(self):
        model = HexWarGNN()
        assert model.hidden_dim == 128
        assert model.history_k == 1

    def test_custom_params(self):
        model = HexWarGNN(hidden_dim=64, n_layers=2, n_heads=2, history_k=3)
        assert model.hidden_dim == 64
        assert model.history_k == 3

    def test_n_layers_matches_conv_layers(self):
        model = HexWarGNN(n_layers=3)
        assert len(model.conv_layers) == 3
        assert len(model.layer_norms) == 3

    def test_node_encoder_input_dim_changes_with_history(self):
        k1 = HexWarGNN(history_k=1)
        k5 = HexWarGNN(history_k=5)
        # node_feature_dim(k) = 18 + (k-1)*4
        assert node_feature_dim(1) != node_feature_dim(5)
        assert k1.node_encoder[0].in_features == node_feature_dim(1)
        assert k5.node_encoder[0].in_features == node_feature_dim(5)


class TestHexWarGNNForward:
    def test_output_shapes_single_graph(self):
        model = HexWarGNN(hidden_dim=32, n_layers=2, n_heads=2)
        model.eval()
        n_nodes, n_edges = 10, 15
        x, edge_index, edge_attr, u, acting_mask = _make_batch(n_nodes, n_edges)

        with torch.no_grad():
            move_logits, alpha, beta, value = model(x, edge_index, edge_attr, u, acting_mask)

        assert move_logits.shape == (n_edges,), f"Expected ({n_edges},) got {move_logits.shape}"
        assert alpha.shape == (n_edges,)
        assert beta.shape == (n_edges,)
        assert value.shape == (1,)

    def test_alpha_beta_positive(self):
        """Beta distribution params must be strictly positive."""
        model = HexWarGNN(hidden_dim=32, n_layers=2, n_heads=2)
        model.eval()
        x, edge_index, edge_attr, u, acting_mask = _make_batch(8, 12)

        with torch.no_grad():
            _, alpha, beta, _ = model(x, edge_index, edge_attr, u, acting_mask)

        assert (alpha > 0).all(), "alpha must be positive"
        assert (beta > 0).all(), "beta must be positive"

    def test_value_is_scalar_per_graph(self):
        """Value head returns one scalar per graph in the batch."""
        model = HexWarGNN(hidden_dim=32, n_layers=2, n_heads=2)
        model.eval()
        x, edge_index, edge_attr, u, acting_mask = _make_batch(6, 10)

        with torch.no_grad():
            _, _, _, value = model(x, edge_index, edge_attr, u, acting_mask)

        assert value.numel() == 1

    def test_no_nan_in_outputs(self):
        """Forward pass must not produce NaN values."""
        model = HexWarGNN(hidden_dim=32, n_layers=2, n_heads=2)
        model.eval()
        x, edge_index, edge_attr, u, acting_mask = _make_batch()

        with torch.no_grad():
            move_logits, alpha, beta, value = model(x, edge_index, edge_attr, u, acting_mask)

        assert not torch.isnan(move_logits).any()
        assert not torch.isnan(alpha).any()
        assert not torch.isnan(beta).any()
        assert not torch.isnan(value).any()

    def test_all_acting_mask_false_no_crash(self):
        """Forward pass with no acting nodes must not crash."""
        model = HexWarGNN(hidden_dim=32, n_layers=2, n_heads=2)
        model.eval()
        x, edge_index, edge_attr, u, acting_mask = _make_batch(6, 8)
        acting_mask[:] = False   # no acting tiles

        with torch.no_grad():
            move_logits, alpha, beta, value = model(x, edge_index, edge_attr, u, acting_mask)

        assert not torch.isnan(value).any()

    def test_history_k5_forward(self):
        """Model with frame stacking (k=5) runs without error."""
        model = HexWarGNN(hidden_dim=32, n_layers=2, n_heads=2, history_k=5)
        model.eval()
        x, edge_index, edge_attr, u, acting_mask = _make_batch(8, 10, history_k=5)

        with torch.no_grad():
            move_logits, alpha, beta, value = model(x, edge_index, edge_attr, u, acting_mask)

        assert move_logits.shape == (10,)

    def test_training_vs_eval_mode(self):
        """Model runs in both training and eval mode."""
        model = HexWarGNN(hidden_dim=32, n_layers=2, n_heads=2)
        x, edge_index, edge_attr, u, acting_mask = _make_batch(6, 8)

        model.train()
        logits_train, _, _, _ = model(x, edge_index, edge_attr, u, acting_mask)

        model.eval()
        with torch.no_grad():
            logits_eval, _, _, _ = model(x, edge_index, edge_attr, u, acting_mask)

        assert logits_train.shape == logits_eval.shape
