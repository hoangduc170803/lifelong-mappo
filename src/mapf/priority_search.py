"""Multi-order prioritized planning baseline.

This is intentionally lightweight: it searches a bounded set of priority
orders and delegates each order to the graph-native prioritized space-time A*
planner. It is a practical Sprint 2 baseline for the warehouse graph, not a
full PBS constraint-tree implementation.
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from typing import Mapping, Optional, Sequence

import networkx as nx

from src.mapf.cbs_solver import MAPFPlanResult, PrioritizedGraphPlanner


class PrioritySearchPlanner:
    """Try multiple priority orders around prioritized graph A*."""

    def __init__(
        self,
        G: nx.DiGraph,
        max_time: int = 512,
        check_following: bool = False,
        max_orders: int = 32,
        reserve_goal_forever: bool = True,
    ):
        if max_orders < 1:
            raise ValueError("max_orders must be >= 1")
        self.G = G
        self.max_time = max_time
        self.check_following = check_following
        self.max_orders = max_orders
        self._planner = PrioritizedGraphPlanner(
            G,
            max_time=max_time,
            check_following=check_following,
            reserve_goal_forever=reserve_goal_forever,
        )

    def plan(
        self,
        starts: Mapping[str, str],
        goals: Mapping[str, str],
        seed: int = 0,
        priority_orders: Optional[Sequence[Sequence[str]]] = None,
    ) -> MAPFPlanResult:
        """Return the first successful priority-order plan, if any."""
        start_time = time.perf_counter()
        orders = (
            self._dedupe_orders(priority_orders)
            if priority_orders is not None
            else self._candidate_orders(starts, goals, seed)
        )

        best_failure: Optional[MAPFPlanResult] = None
        failures: list[dict[str, object]] = []
        for idx, order in enumerate(orders, start=1):
            result = self._planner.plan(starts, goals, priority_order=order)
            if result.success:
                result.solver = "priority_search"
                result.elapsed_s = time.perf_counter() - start_time
                result.diagnostics.update(
                    {
                        "base_solver": "prioritized_graph_astar",
                        "priority_order": list(order),
                        "orders_tried": idx,
                    }
                )
                return result

            failures.append(
                {
                    "priority_order": list(order),
                    "failed_agent": result.diagnostics.get("failed_agent"),
                    "failure_subtype": result.diagnostics.get("failure_subtype"),
                    "planned_agents": len(result.paths),
                }
            )
            if best_failure is None or len(result.paths) > len(best_failure.paths):
                best_failure = result

        elapsed = time.perf_counter() - start_time
        subtype_counts = Counter(
            f["failure_subtype"] for f in failures if f.get("failure_subtype")
        )
        dominant_subtype = (
            subtype_counts.most_common(1)[0][0]
            if subtype_counts
            else "prioritized_block"
        )
        return MAPFPlanResult(
            success=False,
            paths=best_failure.paths if best_failure is not None else {},
            solver="priority_search",
            elapsed_s=elapsed,
            diagnostics={
                "base_solver": "prioritized_graph_astar",
                "orders_tried": len(orders),
                "failures": failures[:10],
                "failure_subtype": dominant_subtype,
                "failure_subtype_counts": dict(subtype_counts),
            },
        )

    def _candidate_orders(
        self,
        starts: Mapping[str, str],
        goals: Mapping[str, str],
        seed: int,
    ) -> list[tuple[str, ...]]:
        agents = tuple(starts)
        if not agents:
            return [agents]

        distances = {
            agent: self._shortest_path_length(starts[agent], goals.get(agent, starts[agent]))
            for agent in agents
        }
        orders: list[Sequence[str]] = [
            agents,
            sorted(agents),
            sorted(agents, key=lambda a: (-distances[a], a)),
            sorted(agents, key=lambda a: (distances[a], a)),
        ]

        for offset in range(1, len(agents)):
            orders.append(agents[offset:] + agents[:offset])

        rng = random.Random(seed)
        while len(orders) < self.max_orders:
            shuffled = list(agents)
            rng.shuffle(shuffled)
            orders.append(shuffled)

        return self._dedupe_orders(orders)[: self.max_orders]

    def _shortest_path_length(self, source: str, target: str) -> float:
        try:
            return float(nx.shortest_path_length(self.G, source, target, weight="weight"))
        except (nx.NetworkXException, KeyError):
            return math.inf

    @staticmethod
    def _dedupe_orders(orders: Sequence[Sequence[str]]) -> list[tuple[str, ...]]:
        out: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()
        for order in orders:
            key = tuple(order)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out
