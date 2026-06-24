"""Tests for the windowed PBS solver (faithful RHCR low-level)."""

from __future__ import annotations

import unittest

import networkx as nx

from src.mapf.cbs_solver import validate_paths
from src.mapf.windowed_pbs import WindowedPBSPlanner


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


def _line_graph(n: int) -> nx.DiGraph:
    G = nx.DiGraph()
    for x in range(n):
        G.add_node(str(x), x=float(x), y=0.0)
    for x in range(n - 1):
        G.add_edge(str(x), str(x + 1), weight=1.0, length=1.0)
        G.add_edge(str(x + 1), str(x), weight=1.0, length=1.0)
    return G


class TestWindowedPBSPlanner(unittest.TestCase):
    def test_head_on_resolves_within_window(self):
        G = _grid_graph(3, 3)
        planner = WindowedPBSPlanner(G, window=16, max_orders=8)
        result = planner.plan(
            starts={"a0": "0,0", "a1": "2,2"},
            goals={"a0": "2,2", "a1": "0,0"},
        )
        self.assertTrue(result.success, result.diagnostics)
        self.assertEqual(result.solver, "windowed_pbs")
        # Paths are real, conflict-free graph transitions.
        self.assertEqual(validate_paths(G, result.paths), [])
        # Window is generous, so both actually reach their goals.
        self.assertEqual(result.paths["a0"][-1], "2,2")
        self.assertEqual(result.paths["a1"][-1], "0,0")

    def test_returns_windowed_prefix_when_goal_is_far(self):
        """The faithful difference vs the full-horizon planner: a goal beyond the
        window yields a *progress* prefix, not a failure."""
        G = _line_graph(10)
        planner = WindowedPBSPlanner(G, window=3)
        result = planner.plan(starts={"a0": "0"}, goals={"a0": "9"})
        self.assertTrue(result.success, result.diagnostics)
        path = result.paths["a0"]
        self.assertLessEqual(len(path), 3 + 1)        # bounded by the window
        self.assertNotEqual(path[-1], "9")            # did NOT need to reach goal
        self.assertEqual(path, ["0", "1", "2", "3"])  # but made max progress
        self.assertEqual(validate_paths(G, result.paths), [])

    def test_shared_goal_needs_no_dedup(self):
        """Two agents dispatched to the SAME node — the case that forced the
        LaCAM reference to de-duplicate waypoints. Windowed PBS handles it: the
        later agent queues behind whoever holds the goal, no dedup, no failure."""
        G = _grid_graph(3, 3)
        planner = WindowedPBSPlanner(G, window=16)
        result = planner.plan(
            starts={"a0": "0,0", "a1": "2,2"},
            goals={"a0": "1,1", "a1": "1,1"},
        )
        self.assertTrue(result.success, result.diagnostics)
        self.assertEqual(validate_paths(G, result.paths), [])
        ends = {result.paths["a0"][-1], result.paths["a1"][-1]}
        self.assertIn("1,1", ends)            # one agent holds the shared goal
        self.assertEqual(len(ends), 2)        # the other waits elsewhere (no overlap)

    def test_idle_agent_holds_when_already_at_goal(self):
        G = _grid_graph(2, 2)
        planner = WindowedPBSPlanner(G, window=8)
        result = planner.plan(starts={"a0": "0,0"}, goals={"a0": "0,0"})
        self.assertTrue(result.success, result.diagnostics)
        self.assertEqual(result.paths["a0"], ["0,0"])

    def test_window_must_be_positive(self):
        with self.assertRaises(ValueError):
            WindowedPBSPlanner(_grid_graph(2, 2), window=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
