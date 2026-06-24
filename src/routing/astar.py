"""Shortest-path routing on the warehouse graph.

Two modes:
  - Lazy A* (`precompute=False`): networkx A* with Euclidean heuristic, paths
    cached as encountered. Good for one-off queries and small experiments.
  - Pre-computed all-pairs (`precompute=True`): scipy.sparse.csgraph.dijkstra
    builds dist[N,N] and next_hop[N,N] matrices in one C-level pass. For the
    1313-node largest SCC of the warehouse, memory is ~21 MB and lookups are
    O(1). Use this for training where features() is called every step.

Both modes expose the same `path`, `distance`, `features` API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class AStarFeatures:
    """A* reference features injected into per-agent observation."""

    next_node_hint: str
    dist_to_goal: float
    path_length_steps: int
    reachable: bool

    @classmethod
    def unreachable(cls, current: str) -> "AStarFeatures":
        return cls(
            next_node_hint=current,
            dist_to_goal=float("inf"),
            path_length_steps=-1,
            reachable=False,
        )


class AStarRouter:
    """Single-pair routing with optional pre-computed all-pairs tables.

    When `precompute=True`, builds two N×N matrices at construction:
      - `_dist_matrix[i,j]` — shortest weighted distance node_i → node_j
      - `_next_hop_matrix[i,j]` — index of the first hop on i → j path
    Lookups become O(1) array indexing.
    """

    UNREACHABLE = -1

    def __init__(self, G: nx.DiGraph, precompute: bool = False):
        self.G = G
        self._heuristic = self._make_euclidean_heuristic()
        self._path_cache: dict[tuple[str, str], list[str]] = {}

        self._dist_matrix: Optional[np.ndarray] = None
        self._next_hop_matrix: Optional[np.ndarray] = None
        self._nodes_list: list[str] = []
        self._node_to_idx: dict[str, int] = {}

        if precompute:
            self._build_all_pairs()

    def _make_euclidean_heuristic(self):
        pos = {n: (data["x"], data["y"]) for n, data in self.G.nodes(data=True)}

        def h(u, v):
            ux, uy = pos[u]
            vx, vy = pos[v]
            return math.hypot(ux - vx, uy - vy)

        return h

    def _build_all_pairs(self):
        """Run scipy Dijkstra from every source; build dist + next_hop matrices."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra

        nodes = list(self.G.nodes())
        n = len(nodes)
        self._nodes_list = nodes
        self._node_to_idx = {node: i for i, node in enumerate(nodes)}

        rows, cols, data = [], [], []
        for u, v, attrs in self.G.edges(data=True):
            rows.append(self._node_to_idx[u])
            cols.append(self._node_to_idx[v])
            data.append(attrs.get("weight", 1.0))
        csr = csr_matrix((data, (rows, cols)), shape=(n, n))

        dist, pred = dijkstra(
            csr, directed=True, return_predecessors=True
        )
        self._dist_matrix = dist  # float64 (N, N), inf if unreachable

        # next_hop[s, t] = first hop on shortest path s → t.
        # Built bottom-up by processing targets in increasing distance from s:
        # if pred[s, t] == s, next_hop is t; otherwise inherit from pred[s, t].
        next_hop = np.full((n, n), self.UNREACHABLE, dtype=np.int32)
        for s in range(n):
            next_hop[s, s] = s
            order = np.argsort(dist[s], kind="stable")
            for t in order:
                if t == s:
                    continue
                p = pred[s, t]
                if p < 0:  # unreachable (scipy sentinel is -9999)
                    next_hop[s, t] = self.UNREACHABLE
                elif p == s:
                    next_hop[s, t] = t
                else:
                    next_hop[s, t] = next_hop[s, p]
        self._next_hop_matrix = next_hop

    def _is_precomputed(self) -> bool:
        return self._dist_matrix is not None

    def path(self, source: str, target: str) -> Optional[list[str]]:
        """Return list of node ids from source to target, or None if unreachable."""
        if source == target:
            return [source]
        if self._is_precomputed():
            si, ti = self._node_to_idx[source], self._node_to_idx[target]
            if not math.isfinite(self._dist_matrix[si, ti]):
                return None
            p: list[str] = [source]
            cur = si
            while cur != ti:
                nxt = int(self._next_hop_matrix[cur, ti])
                if nxt < 0:
                    return None
                p.append(self._nodes_list[nxt])
                cur = nxt
            return p

        key = (source, target)
        cached = self._path_cache.get(key)
        if cached is not None:
            return cached
        try:
            p = nx.astar_path(
                self.G, source, target, heuristic=self._heuristic, weight="weight"
            )
        except nx.NetworkXNoPath:
            return None
        self._path_cache[key] = p
        return p

    def distance(self, source: str, target: str) -> float:
        """Shortest path distance using edge weights. inf if unreachable."""
        if source == target:
            return 0.0
        if self._is_precomputed():
            si, ti = self._node_to_idx[source], self._node_to_idx[target]
            return float(self._dist_matrix[si, ti])
        p = self.path(source, target)
        if p is None:
            return float("inf")
        return sum(
            self.G.edges[p[i], p[i + 1]]["weight"] for i in range(len(p) - 1)
        )

    def next_hop(self, source: str, target: str) -> Optional[str]:
        """First node on shortest path source → target. None if unreachable."""
        if source == target:
            return source
        if self._is_precomputed():
            si, ti = self._node_to_idx[source], self._node_to_idx[target]
            idx = int(self._next_hop_matrix[si, ti])
            if idx == self.UNREACHABLE:
                return None
            return self._nodes_list[idx]
        p = self.path(source, target)
        return p[1] if p is not None and len(p) > 1 else None

    def features(self, current: str, goal: str) -> AStarFeatures:
        """Build A* features for one agent at `current` heading to `goal`."""
        if current == goal:
            return AStarFeatures(
                next_node_hint=current,
                dist_to_goal=0.0,
                path_length_steps=0,
                reachable=True,
            )
        if self._is_precomputed():
            si, ti = self._node_to_idx[current], self._node_to_idx[goal]
            d = float(self._dist_matrix[si, ti])
            if not math.isfinite(d):
                return AStarFeatures.unreachable(current)
            next_idx = int(self._next_hop_matrix[si, ti])
            if next_idx == self.UNREACHABLE:
                return AStarFeatures.unreachable(current)
            # path length in steps is unknown O(1); not needed for the env, skip.
            return AStarFeatures(
                next_node_hint=self._nodes_list[next_idx],
                dist_to_goal=d,
                path_length_steps=-1,
                reachable=True,
            )

        p = self.path(current, goal)
        if p is None:
            return AStarFeatures.unreachable(current)
        return AStarFeatures(
            next_node_hint=p[1],
            dist_to_goal=self.distance(current, goal),
            path_length_steps=len(p) - 1,
            reachable=True,
        )

    def clear_cache(self):
        self._path_cache.clear()


if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    from map_parser import parse_opentcs_map

    G = parse_opentcs_map(
        str(repo_root / "warehouse_map.xml"),
        restrict_to_largest_scc=True,
    )
    print(f"Graph (largest SCC): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    t0 = time.perf_counter()
    router = AStarRouter(G, precompute=True)
    print(f"All-pairs precompute: {time.perf_counter() - t0:.2f}s")

    dist = router._dist_matrix
    n_reachable = int(np.isfinite(dist).sum())
    print(f"Reachable pairs: {n_reachable}/{dist.size} "
          f"({n_reachable / dist.size * 100:.2f}%)")
    print(f"Memory: dist={dist.nbytes/1e6:.1f}MB, "
          f"next_hop={router._next_hop_matrix.nbytes/1e6:.1f}MB")

    nodes = list(G.nodes())
    src, dst = nodes[0], nodes[len(nodes) // 2]

    t0 = time.perf_counter()
    for _ in range(10000):
        router.features(src, dst)
    print(f"10k features() calls: {(time.perf_counter() - t0) * 1000:.1f}ms")

    print(f"\nA* test: {src} -> {dst}")
    feat = router.features(src, dst)
    print(f"  Features: {feat}")
    path = router.path(src, dst)
    print(f"  Path: {len(path)} nodes, "
          f"reconstructed first/last: {path[0]} ... {path[-1]}")
