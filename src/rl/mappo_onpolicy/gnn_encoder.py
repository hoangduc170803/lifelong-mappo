"""GNN observation encoder for warehouse MAPPO (Sprint 4b).

Drop-in replacement for on-policy's `MLPBase`: same constructor signature
``(args, obs_shape)`` and same ``forward(obs) -> (B, hidden_size)`` contract,
so `R_Actor`/`R_Critic` can swap it in by reading ``args.encoder``.

The per-agent observation is a flat vector laid out by
`WarehouseEnv._build_obs` as:

    [ self_x, self_y, gx, gy, d_norm,          # 5  ego kinematics
      astar_oh (NUM_ACTIONS = 9),              # 9  A* hint one-hot
      state_oh (3),                            # 3  agent FSM state
      knn (3 * knn_agents) ]                   # k blocks of (dx, dy, has_task)

The k neighbour blocks form a star graph: each neighbour agent is a node with
edges INTO the ego node. A `GATv2Conv` lets the ego attend over its neighbours
(learned, permutation-invariant pooling) instead of the MLP's fixed
concatenation. The attended agent-context is concatenated with the ego block
and projected to ``hidden_size`` through the same Tanh/LayerNorm MLP stack the
rest of the policy uses.

NOTE (scope): the current env observation carries the agent kNN graph only,
not a local map subgraph, so this encoder is GAT-over-agents. The
"GCN-over-map-subgraph" half of the planned architecture needs the env to emit
local node features + edges (a separate obs extension) and is deferred; see
PLAN Sprint 4b.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from onpolicy.algorithms.utils.mlp import MLPLayer

# Mirror of the env constants used to slice the flat observation.
_EGO_DIM = 5  # self_x, self_y, gx, gy, d_norm
_NUM_ACTIONS = 9  # compass_mapper.NUM_ACTIONS (astar hint one-hot)
_STATE_DIM = 3  # agent FSM one-hot
_NEIGHBOR_FEAT = 3  # (dx, dy, has_task) per neighbour
_EGO_BLOCK = _EGO_DIM + _NUM_ACTIONS + _STATE_DIM  # 17


class GNNBase(nn.Module):
    """GAT-over-agents encoder with the MLPBase interface."""

    def __init__(self, args, obs_shape):
        super().__init__()
        self._use_feature_normalization = args.use_feature_normalization
        self.hidden_size = args.hidden_size
        self._knn = int(getattr(args, "knn_agents", 3))
        self._gat_heads = int(getattr(args, "gat_heads", 4))
        node_dim = int(getattr(args, "gnn_node_dim", 32))

        obs_dim = int(obs_shape[0])
        expected = _EGO_BLOCK + _NEIGHBOR_FEAT * self._knn
        if obs_dim != expected:
            raise ValueError(
                f"GNNBase expects obs_dim {expected} for knn={self._knn} "
                f"(ego block {_EGO_BLOCK} + {_NEIGHBOR_FEAT}*{self._knn}); "
                f"got {obs_dim}. Pass --knn_agents matching the env."
            )

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        # Import lazily so a CPU-only / no-PyG environment can still import the
        # module (the error only fires if the GNN encoder is actually selected).
        from torch_geometric.nn import GATv2Conv

        self.ego_proj = nn.Linear(_EGO_BLOCK, node_dim)
        self.neighbor_proj = nn.Linear(_NEIGHBOR_FEAT, node_dim)
        self.gat = GATv2Conv(
            node_dim, node_dim, heads=self._gat_heads, concat=False, add_self_loops=False
        )
        self.act = nn.Tanh()

        # Ego embedding ++ attended agent-context -> hidden_size, then the same
        # Tanh/LayerNorm MLP stack used by MLPBase for parity.
        self.mlp = MLPLayer(
            _EGO_BLOCK + node_dim,
            self.hidden_size,
            args.layer_N,
            args.use_orthogonal,
            args.use_ReLU,
        )

        # Per-(batch, device) cached star edge_index: neighbours -> ego.
        self._edge_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    def _edge_index(self, batch: int, device: torch.device) -> torch.Tensor:
        key = (batch, device)
        cached = self._edge_cache.get(key)
        if cached is not None:
            return cached
        nodes_per = self._knn + 1  # ego + k neighbours
        srcs: list[int] = []
        dsts: list[int] = []
        for b in range(batch):
            ego = b * nodes_per
            for j in range(self._knn):
                srcs.append(ego + 1 + j)  # neighbour node
                dsts.append(ego)  # -> ego
        edge_index = torch.tensor([srcs, dsts], dtype=torch.long, device=device)
        self._edge_cache[key] = edge_index
        return edge_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        ego_block = x[:, :_EGO_BLOCK]
        neighbors = x[:, _EGO_BLOCK:].reshape(-1, self._knn, _NEIGHBOR_FEAT)
        batch = x.shape[0]

        ego_node = self.act(self.ego_proj(ego_block))  # (B, node_dim)
        nbr_nodes = self.act(self.neighbor_proj(neighbors))  # (B, k, node_dim)

        # Interleave into a flat node list of (ego, n1..nk) per sample.
        nodes = torch.cat(
            [ego_node.unsqueeze(1), nbr_nodes], dim=1
        ).reshape(batch * (self._knn + 1), -1)
        edge_index = self._edge_index(batch, x.device)
        attended = self.gat(nodes, edge_index)  # (B*(k+1), node_dim)

        nodes_per = self._knn + 1
        ego_idx = torch.arange(batch, device=x.device) * nodes_per
        agent_context = attended[ego_idx]  # (B, node_dim)

        fused = torch.cat([ego_block, agent_context], dim=1)
        return self.mlp(fused)
