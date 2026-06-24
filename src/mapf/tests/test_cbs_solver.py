"""Tests for graph-safe MAPF planning and path validation."""

from __future__ import annotations

import importlib.util
import unittest

import networkx as nx

from src.mapf.cbs_solver import CBSMapfPlanner, PrioritizedGraphPlanner, validate_paths


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


class TestPathValidation(unittest.TestCase):
    def setUp(self):
        self.G = _grid_graph(2, 2)

    def test_detects_vertex_conflict(self):
        conflicts = validate_paths(
            self.G,
            {
                "a0": ["0,0", "0,1"],
                "a1": ["1,1", "0,1"],
            },
        )
        self.assertTrue(any(c.type == "vertex" for c in conflicts))

    def test_detects_edge_swap(self):
        conflicts = validate_paths(
            self.G,
            {
                "a0": ["0,0", "0,1"],
                "a1": ["0,1", "0,0"],
            },
        )
        self.assertTrue(any(c.type == "edge_swap" for c in conflicts))

    def test_detects_invalid_transition(self):
        conflicts = validate_paths(self.G, {"a0": ["0,0", "1,1"]})
        self.assertTrue(any(c.type == "invalid_transition" for c in conflicts))


class TestPrioritizedGraphPlanner(unittest.TestCase):
    def test_plans_collision_free_paths_on_graph(self):
        G = _grid_graph(3, 3)
        planner = PrioritizedGraphPlanner(G, max_time=16)
        result = planner.plan(
            starts={"a0": "0,0", "a1": "2,2"},
            goals={"a0": "2,2", "a1": "0,0"},
        )
        self.assertTrue(result.success, result)
        self.assertEqual(result.paths["a0"][0], "0,0")
        self.assertEqual(result.paths["a0"][-1], "2,2")
        self.assertEqual(result.paths["a1"][0], "2,2")
        self.assertEqual(result.paths["a1"][-1], "0,0")
        self.assertEqual(validate_paths(G, result.paths), [])

    def test_adapter_fallback_runs_without_external_package(self):
        G = _grid_graph(3, 3)
        planner = CBSMapfPlanner(G, max_time=16)
        result = planner.plan(
            starts={"a0": "0,0", "a1": "2,2"},
            goals={"a0": "2,2", "a1": "0,0"},
            use_external=False,
        )
        self.assertTrue(result.success, result)
        self.assertEqual(result.solver, "prioritized_graph_astar")

    def test_reports_no_path_failure_subtype(self):
        G = nx.DiGraph()
        G.add_node("a", x=0.0, y=0.0, is_halt=False, type="POINT")
        G.add_node("b", x=1.0, y=0.0, is_halt=False, type="POINT")
        G.add_edge("a", "b", weight=1.0, length=1.0)
        result = PrioritizedGraphPlanner(G, max_time=4).plan(
            starts={"a0": "b"},
            goals={"a0": "a"},
        )
        self.assertFalse(result.success)
        self.assertEqual(result.diagnostics["failure_subtype"], "no_path")

    def test_reports_prioritized_block_failure_subtype(self):
        G = _grid_graph(1, 2)
        result = PrioritizedGraphPlanner(G, max_time=4).plan(
            starts={"a0": "0,0", "a1": "0,1"},
            goals={"a0": "0,1", "a1": "0,0"},
        )
        self.assertFalse(result.success)
        self.assertEqual(result.diagnostics["failure_subtype"], "prioritized_block")

    def test_coord_collisions_are_reported_in_diagnostics(self):
        G = nx.DiGraph()
        G.add_node("main", x=0.0, y=0.0, is_halt=False, type="POINT")
        G.add_node("halt", x=0.0, y=0.0, is_halt=True, type="HALT_POSITION")
        G.add_node("goal", x=1.0, y=0.0, is_halt=False, type="POINT")
        G.add_edge("halt", "goal", weight=1.0, length=1.0)

        planner = CBSMapfPlanner(G, max_time=8)
        diag = planner.compact_grid_diagnostics()
        self.assertEqual(diag["coord_collision_nodes"], 1)
        self.assertEqual(diag["coord_collision_samples"][0]["node"], "halt")

        result = planner.plan(
            starts={"a0": "halt"},
            goals={"a0": "goal"},
            fallback_on_failure=False,
        )
        self.assertFalse(result.success)
        self.assertEqual(result.diagnostics["coord_collision_nodes"], 1)
        self.assertEqual(result.diagnostics["error"], "node has no unique compact coordinate")

    @unittest.skipUnless(
        importlib.util.find_spec("cbs_mapf"),
        "cbs-mapf not installed",
    )
    def test_external_cbs_mapf_actually_works(self):
        G = _grid_graph(3, 3)
        planner = CBSMapfPlanner(G, max_time=16)
        # Use a straight-line case so cbs-mapf's 8-neighbor grid planner does
        # not take a diagonal shortcut that the directed graph would reject.
        result = planner.plan(
            starts={"a0": "0,0"},
            goals={"a0": "0,2"},
            fallback_on_failure=False,
        )
        self.assertTrue(result.success, result.diagnostics)
        self.assertEqual(result.solver, "cbs_mapf", result.diagnostics)
        self.assertEqual(result.paths["a0"], ["0,0", "0,1", "0,2"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
