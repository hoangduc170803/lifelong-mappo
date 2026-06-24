"""Tests for the Sprint 4b GAT observation encoder."""

from __future__ import annotations

import types
import unittest

import torch

from src.rl.mappo_onpolicy.gnn_encoder import GNNBase, _EGO_BLOCK, _NEIGHBOR_FEAT


def _args(knn=3, hidden=64):
    return types.SimpleNamespace(
        use_feature_normalization=True,
        hidden_size=hidden,
        knn_agents=knn,
        gat_heads=4,
        gnn_node_dim=32,
        layer_N=2,
        use_orthogonal=True,
        use_ReLU=False,
    )


class TestGNNBase(unittest.TestCase):
    def test_forward_shape_matches_hidden_size(self):
        knn = 3
        obs_dim = _EGO_BLOCK + _NEIGHBOR_FEAT * knn
        base = GNNBase(_args(knn), (obs_dim,))
        out = base(torch.randn(16, obs_dim))
        self.assertEqual(tuple(out.shape), (16, 64))
        self.assertTrue(torch.isfinite(out).all())

    def test_obs_dim_mismatch_raises(self):
        # knn=3 expects 26; pass a 30-d obs -> clear error, not silent slice.
        with self.assertRaises(ValueError):
            GNNBase(_args(knn=3), (30,))

    def test_gradients_flow_to_all_params(self):
        knn = 2
        obs_dim = _EGO_BLOCK + _NEIGHBOR_FEAT * knn
        base = GNNBase(_args(knn), (obs_dim,))
        out = base(torch.randn(8, obs_dim))
        out.sum().backward()
        for name, p in base.named_parameters():
            self.assertIsNotNone(p.grad, f"{name} got no gradient")
            self.assertTrue(torch.isfinite(p.grad).all(), f"{name} non-finite grad")

    def test_permutation_invariance_over_neighbors(self):
        """Reordering the k neighbour blocks must not change the output:
        the GAT pools neighbours as a set, unlike the MLP's fixed concat."""
        knn = 3
        obs_dim = _EGO_BLOCK + _NEIGHBOR_FEAT * knn
        base = GNNBase(_args(knn), (obs_dim,)).eval()
        ego = torch.randn(1, _EGO_BLOCK)
        nbrs = torch.randn(1, knn, _NEIGHBOR_FEAT)
        x1 = torch.cat([ego, nbrs.reshape(1, -1)], dim=1)
        perm = torch.tensor([2, 0, 1])
        x2 = torch.cat([ego, nbrs[:, perm, :].reshape(1, -1)], dim=1)
        with torch.no_grad():
            y1, y2 = base(x1), base(x2)
        self.assertTrue(torch.allclose(y1, y2, atol=1e-5))

    def test_batch_edge_index_cached_and_correct_size(self):
        knn = 3
        obs_dim = _EGO_BLOCK + _NEIGHBOR_FEAT * knn
        base = GNNBase(_args(knn), (obs_dim,))
        ei = base._edge_index(5, torch.device("cpu"))
        # 5 samples * knn neighbour->ego edges.
        self.assertEqual(ei.shape, (2, 5 * knn))
        # Cached object identity on second call.
        self.assertIs(ei, base._edge_index(5, torch.device("cpu")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
