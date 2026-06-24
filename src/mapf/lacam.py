"""LaCAM — Lazy Constraints Addition search for MAPF (Okumura, AAAI 2023).

One-shot MAPF solver: depth-first search over joint *configurations*, using
PIBT as the configuration generator. Each high-level node lazily enumerates
low-level constraints ("agent k must go to vertex v") so that, when the
unconstrained PIBT successor leads nowhere new, alternative joint moves are
explored instead of giving up — this is what lets LaCAM escape the local
minima that trap plain iterated PIBT (e.g., two agents that must pass
through a siding).

Python port of the algorithmic core of Kei18/lacam. Faithful to the canonical
two-level structure: peek-then-pop configuration stack; lazy agent-by-agent
constraint tree of neighbours+stay; PIBT as the configuration generator (note
the author's teaching port `kei18/pylacam` substitutes a *random* generator, so
this is closer to the published LaCAM on that axis); and — the completeness
mechanism — re-insertion of an already-explored configuration's node back onto
OPEN (canonical LaCAM / pylacam `OPEN.appendleft(N_known)`; here
`open_stack.append(known)`, since our stack top is the right end). With the
re-insertion the search retains LaCAM's completeness over the reachable
configuration space rather than being a one-shot depth-first variant. The only
deliberate simplification is no LaCAM* anytime cost refinement (we need
feasibility for the windowed sub-instances, not an optimised sum-of-costs), and
the search is bounded by `max_expansions` / `max_time_s`. Serves as a
deliberative classical reference alongside the faithful windowed RHCR/PBS strong
bar (`src/mapf/windowed_pbs.py`). Sibling to `src/mapf/priority_search.py` (PP-32).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

import networkx as nx

from src.mapf.cbs_solver import MAPFPlanResult
from src.mapf.pibt import PIBT


@dataclass
class _HLNode:
    """High-level search node: one joint configuration."""

    config: tuple[str, ...]
    parent: Optional["_HLNode"]
    depth: int
    # Low-level constraint tree, enumerated lazily. Each entry is a list of
    # (agent, vertex) pairs that PIBT must respect when generating the
    # successor configuration. The first entry is always [] (unconstrained).
    tree: deque[list[tuple[str, str]]] = field(default_factory=lambda: deque([[]]))
    tree_expanded_upto: int = 0  # how many agents already have branches


class LaCAMPlanner:
    """Minimal LaCAM for one-shot MAPF on a (directed) graph."""

    def __init__(
        self,
        G: nx.DiGraph,
        dist: Optional[Callable[[str, str], float]] = None,
        seed: int = 0,
        max_expansions: int = 30_000,
        max_time_s: float = 30.0,
    ):
        self.G = G
        if dist is None:
            lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))

            def dist(u: str, v: str) -> float:  # type: ignore[misc]
                return float(lengths[u].get(v, float("inf")))

        self.dist = dist
        self.seed = seed
        self.max_expansions = max_expansions
        self.max_time_s = max_time_s
        self._pibt = PIBT(G, dist, seed=seed)

    # ------------------------------------------------------------------ API

    def plan(
        self,
        starts: Mapping[str, str],
        goals: Mapping[str, str],
    ) -> MAPFPlanResult:
        t0 = time.perf_counter()
        agents = sorted(starts)
        goal_config = tuple(goals[a] for a in agents)
        init = _HLNode(config=tuple(starts[a] for a in agents), parent=None, depth=0)

        open_stack: list[_HLNode] = [init]
        explored: dict[tuple[str, ...], _HLNode] = {init.config: init}
        expansions = 0

        while open_stack:
            if (
                expansions >= self.max_expansions
                or time.perf_counter() - t0 > self.max_time_s
            ):
                break
            node = open_stack[-1]
            if node.config == goal_config:
                paths = self._reconstruct(agents, node)
                return MAPFPlanResult(
                    success=True,
                    paths=paths,
                    solver="lacam",
                    makespan=node.depth,
                    sum_of_costs=sum(len(p) - 1 for p in paths.values()),
                    elapsed_s=time.perf_counter() - t0,
                    diagnostics={"expansions": expansions},
                )
            if not node.tree:
                open_stack.pop()
                continue

            constraints = node.tree.popleft()
            self._grow_tree(agents, node, constraints)
            expansions += 1

            positions = {a: v for a, v in zip(agents, node.config)}
            priorities = {
                a: self.dist(positions[a], goals[a]) for a in agents
            }
            moves = self._pibt.step(
                positions, goals, priorities, forced=dict(constraints) or None
            )
            if moves is None:
                continue
            config_new = tuple(moves[a] for a in agents)
            known = explored.get(config_new)
            if known is not None:
                if known is not node:
                    # Canonical LaCAM completeness step (pylacam does
                    # `OPEN.appendleft(N_known)`): bring an already-explored
                    # configuration's node back to the top of OPEN so its
                    # possibly-unexhausted constraint tree is revisited via this
                    # path. Without this the search is an incomplete DFS.
                    open_stack.append(known)
                continue
            child = _HLNode(config=config_new, parent=node, depth=node.depth + 1)
            explored[config_new] = child
            open_stack.append(child)

        return MAPFPlanResult(
            success=False,
            paths={},
            solver="lacam",
            elapsed_s=time.perf_counter() - t0,
            diagnostics={
                "expansions": expansions,
                "failure_subtype": (
                    "search_exhausted" if not open_stack else "budget_exceeded"
                ),
            },
        )

    # ------------------------------------------------------------------ internals

    def _grow_tree(
        self,
        agents: list[str],
        node: _HLNode,
        constraints: list[tuple[str, str]],
    ) -> None:
        """Lazily add the next agent's branches under `constraints`."""
        k = len(constraints)
        if k >= len(agents):
            return
        agent_k = agents[k]
        pos_k = node.config[agents.index(agent_k)]
        candidates = list(self.G.successors(pos_k)) + [pos_k]
        for vertex in candidates:
            node.tree.append(constraints + [(agent_k, vertex)])

    @staticmethod
    def _reconstruct(agents: list[str], node: _HLNode) -> dict[str, list[str]]:
        configs: list[tuple[str, ...]] = []
        cur: Optional[_HLNode] = node
        while cur is not None:
            configs.append(cur.config)
            cur = cur.parent
        configs.reverse()
        return {
            a: [config[i] for config in configs]
            for i, a in enumerate(agents)
        }
