"""Tests for the multi-order priority-search MAPF baseline."""

from __future__ import annotations

import unittest

import networkx as nx

from src.mapf.priority_search import PrioritySearchPlanner
from src.mapf.cbs_solver import validate_paths


def _grid_graph(rows: int, cols: int) -> nx.DiGraph:
    G = nx.DiGraph()
    for r in range(rows):
        for c in range(cols):
            G.add_node(f"{r},{c}", x=float(c), y=float(r), is_halt=False, type="POINT")
    for r in range(rows):
        for c in range(cols):
            here = f"{r},{c}"
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    G.add_edge(here, f"{nr},{nc}", weight=1.0, length=1.0)
    return G


class TestPrioritySearchPlanner(unittest.TestCase):
    def test_finds_collision_free_plan(self):
        G = _grid_graph(3, 3)
        planner = PrioritySearchPlanner(G, max_time=16, max_orders=8)
        result = planner.plan(
            starts={"a0": "0,0", "a1": "2,2"},
            goals={"a0": "2,2", "a1": "0,0"},
            seed=7,
        )
        self.assertTrue(result.success, result.diagnostics)
        self.assertEqual(result.solver, "priority_search")
        self.assertEqual(validate_paths(G, result.paths), [])
        self.assertGreaterEqual(result.diagnostics["orders_tried"], 1)

    def test_reports_failure_after_bounded_orders(self):
        G = _grid_graph(1, 2)
        planner = PrioritySearchPlanner(G, max_time=1, max_orders=2)
        result = planner.plan(
            starts={"a0": "0,0", "a1": "0,1"},
            goals={"a0": "0,1", "a1": "0,0"},
        )
        self.assertFalse(result.success)
        self.assertEqual(result.solver, "priority_search")
        self.assertEqual(result.diagnostics["orders_tried"], 2)

    def test_failure_propagates_subtype_from_underlying_planner(self):
        """Issue #2 fix: PrioritySearchPlanner must expose the dominant
        failure_subtype from underlying prioritized A* attempts so that
        downstream benchmark `_failure_reason` can classify it as
        `prioritized_block` rather than the generic `planner_failure`."""
        G = _grid_graph(1, 2)
        planner = PrioritySearchPlanner(G, max_time=1, max_orders=2)
        result = planner.plan(
            starts={"a0": "0,0", "a1": "0,1"},
            goals={"a0": "0,1", "a1": "0,0"},
        )
        self.assertFalse(result.success)
        self.assertIn("failure_subtype", result.diagnostics)
        self.assertEqual(result.diagnostics["failure_subtype"], "prioritized_block")
        counts = result.diagnostics.get("failure_subtype_counts")
        self.assertIsInstance(counts, dict)
        self.assertEqual(counts.get("prioritized_block"), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
