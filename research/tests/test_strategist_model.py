"""Tests for StrategistGNN architecture and forward pass."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import torch
from hexwar.agents.neural.encoder import N_EDGE_FEATURES, N_GLOBAL_FEATURES, N_NODE_FEATURES
from hexwar.agents.neural.strategist_model import StrategistGNN


def _make_batch(n_nodes: int = 8, n_edges: int = 12):
    x = torch.randn(n_nodes, N_NODE_FEATURES)
    src = torch.randint(0, n_nodes, (n_edges,))
    dst = torch.randint(0, n_nodes, (n_edges,))
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = torch.randn(n_edges, N_EDGE_FEATURES)
    u = torch.randn(N_GLOBAL_FEATURES)
    acting_mask = torch.zeros(n_nodes, dtype=torch.bool)
    acting_mask[:3] = True
    return x, edge_index, edge_attr, u, acting_mask


class TestStrategistGNNInit:
    def test_default_instantiation(self):
        model = StrategistGNN()
        assert model.hidden_dim == 128

    def test_custom_params(self):
        model = StrategistGNN(hidden_dim=64, n_layers=2, n_heads=2)
        assert model.hidden_dim == 64
        assert len(model.conv_layers) == 2

    def test_gru_cell_present(self):
        model = StrategistGNN()
        import torch.nn as nn
        assert isinstance(model.gru_cell, nn.GRUCell)

    def test_global_attn_present(self):
        model = StrategistGNN()
        import torch.nn as nn
        assert isinstance(model.global_attn, nn.MultiheadAttention)

    def test_node_encoder_takes_node_features_plus_one(self):
        model = StrategistGNN(hidden_dim=64)
        assert model.node_encoder[0].in_features == N_NODE_FEATURES + 1


class TestStrategistGNNForward:
    def test_output_shapes(self):
        model = StrategistGNN(hidden_dim=32, n_layers=2, n_heads=2)
        model.eval()
        n_nodes, n_edges = 10, 15
        x, edge_index, edge_attr, u, acting_mask = _make_batch(n_nodes, n_edges)

        with torch.no_grad():
            ml, alpha, beta, value, h_new = model(x, edge_index, edge_attr, u, acting_mask)

        assert ml.shape == (n_edges,)
        assert alpha.shape == (n_edges,)
        assert beta.shape == (n_edges,)
        assert value.shape == (1,)
        assert h_new.shape == (n_nodes, 32)

    def test_returns_five_outputs(self):
        model = StrategistGNN(hidden_dim=32, n_layers=2, n_heads=2)
        model.eval()
        x, edge_index, edge_attr, u, acting_mask = _make_batch()

        with torch.no_grad():
            out = model(x, edge_index, edge_attr, u, acting_mask)

        assert len(out) == 5

    def test_alpha_beta_positive(self):
        model = StrategistGNN(hidden_dim=32, n_layers=2, n_heads=2)
        model.eval()
        x, edge_index, edge_attr, u, acting_mask = _make_batch()

        with torch.no_grad():
            _, alpha, beta, _, _ = model(x, edge_index, edge_attr, u, acting_mask)

        assert (alpha > 0).all()
        assert (beta > 0).all()

    def test_no_nan_in_outputs(self):
        model = StrategistGNN(hidden_dim=32, n_layers=2, n_heads=2)
        model.eval()
        x, edge_index, edge_attr, u, acting_mask = _make_batch()

        with torch.no_grad():
            ml, alpha, beta, value, h_new = model(x, edge_index, edge_attr, u, acting_mask)

        assert not torch.isnan(ml).any()
        assert not torch.isnan(alpha).any()
        assert not torch.isnan(beta).any()
        assert not torch.isnan(value).any()
        assert not torch.isnan(h_new).any()

    def test_h_tiles_none_initialises_to_zeros(self):
        """With h_tiles=None the model initialises GRU state to zeros internally."""
        model = StrategistGNN(hidden_dim=32, n_layers=2, n_heads=2)
        model.eval()
        x, edge_index, edge_attr, u, acting_mask = _make_batch()

        with torch.no_grad():
            _, _, _, _, h_new = model(x, edge_index, edge_attr, u, acting_mask, h_tiles=None)

        assert h_new.shape == (x.size(0), 32)

    def test_h_tiles_passed_in_changes_output(self):
        """Providing different h_tiles produces different outputs (GRU is active)."""
        model = StrategistGNN(hidden_dim=32, n_layers=2, n_heads=2)
        model.eval()
        x, edge_index, edge_attr, u, acting_mask = _make_batch()

        h_zero = torch.zeros(x.size(0), 32)
        h_rand = torch.randn(x.size(0), 32)

        with torch.no_grad():
            ml_zero, _, _, _, _ = model(x, edge_index, edge_attr, u, acting_mask, h_tiles=h_zero)
            ml_rand, _, _, _, _ = model(x, edge_index, edge_attr, u, acting_mask, h_tiles=h_rand)

        assert not torch.allclose(ml_zero, ml_rand)

    def test_turn_frac_changes_output(self):
        """Different turn_frac values produce different outputs (phase awareness)."""
        model = StrategistGNN(hidden_dim=32, n_layers=2, n_heads=2)
        model.eval()
        x, edge_index, edge_attr, u, acting_mask = _make_batch()

        with torch.no_grad():
            ml_early, _, _, _, _ = model(x, edge_index, edge_attr, u, acting_mask, turn_frac=0.0)
            ml_late,  _, _, _, _ = model(x, edge_index, edge_attr, u, acting_mask, turn_frac=1.0)

        assert not torch.allclose(ml_early, ml_late)

    def test_h_tiles_new_detached(self):
        """h_tiles_new must be detached (no grad) for safe external storage."""
        model = StrategistGNN(hidden_dim=32, n_layers=2, n_heads=2)
        x, edge_index, edge_attr, u, acting_mask = _make_batch()

        _, _, _, _, h_new = model(x, edge_index, edge_attr, u, acting_mask)
        assert not h_new.requires_grad

    def test_all_acting_mask_false(self):
        """Forward pass with no acting tiles must not crash."""
        model = StrategistGNN(hidden_dim=32, n_layers=2, n_heads=2)
        model.eval()
        x, edge_index, edge_attr, u, acting_mask = _make_batch(6, 8)
        acting_mask[:] = False

        with torch.no_grad():
            _, _, _, value, _ = model(x, edge_index, edge_attr, u, acting_mask)

        assert not torch.isnan(value).any()
