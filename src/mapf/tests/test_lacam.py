"""Tests for the minimal LaCAM planner (Okumura 2023)."""

from __future__ import annotations

import unittest

import networkx as nx

from src.mapf.cbs_solver import validate_paths
from src.mapf.lacam import LaCAMPlanner


def _bidir(G, u, v):
    G.add_edge(u, v, weight=1.0)
    G.add_edge(v, u, weight=1.0)


def _line(n=4):
    G = nx.DiGraph()
    for x in range(n):
        G.add_node(str(x), x=float(x), y=0.0)
    for x in range(n - 1):
        _bidir(G, str(x), str(x + 1))
    return G


def _corridor_with_siding():
    """0-1-2-3 corridor with a siding S hanging off node 1."""
    G = _line(4)
    G.add_node("S", x=1.0, y=1.0)
    _bidir(G, "1", "S")
    return G


def _grid(w=4, h=4):
    G = nx.DiGraph()
    for x in range(w):
        for y in range(h):
            G.add_node(f"{x},{y}", x=float(x), y=float(y))
    for x in range(w):
        for y in range(h):
            for dx, dy in ((1, 0), (0, 1)):
                nx_, ny_ = x + dx, y + dy
                if nx_ < w and ny_ < h:
                    _bidir(G, f"{x},{y}", f"{nx_},{ny_}")
    return G


def _assert_solution_valid(testcase, G, planner_result, starts, goals):
    testcase.assertTrue(planner_result.success)
    paths = planner_result.paths
    for a, start in starts.items():
        testcase.assertEqual(paths[a][0], start)
        testcase.assertEqual(paths[a][-1], goals[a])
    conflicts = validate_paths(G, paths)
    testcase.assertEqual(conflicts, [], f"conflicts found: {conflicts[:3]}")


class TestLaCAM(unittest.TestCase):
    def test_single_agent_straight_line(self):
        G = _line(4)
        result = LaCAMPlanner(G, seed=0).plan({"a": "0"}, {"a": "3"})
        _assert_solution_valid(self, G, result, {"a": "0"}, {"a": "3"})
        self.assertEqual(result.makespan, 3)

    def test_head_on_corridor_with_siding(self):
        """The classic case plain iterated PIBT cannot solve: two agents in
        a corridor must use the siding to pass. LaCAM's constraint tree
        finds it."""
        G = _corridor_with_siding()
        starts = {"a": "0", "b": "3"}
        goals = {"a": "3", "b": "0"}
        result = LaCAMPlanner(G, seed=0).plan(starts, goals)
        _assert_solution_valid(self, G, result, starts, goals)

    def test_pure_corridor_head_on_is_unsolvable(self):
        """No siding -> physically impossible; LaCAM must report failure,
        not hang or emit a conflicting plan."""
        G = _line(4)
        result = LaCAMPlanner(G, seed=0, max_expansions=2000).plan(
            {"a": "0", "b": "3"}, {"a": "3", "b": "0"}
        )
        self.assertFalse(result.success)

    def test_grid_crossing_agents(self):
        G = _grid(4, 4)
        starts = {"a": "0,0", "b": "3,3", "c": "0,3", "d": "3,0"}
        goals = {"a": "3,3", "b": "0,0", "c": "3,0", "d": "0,3"}
        result = LaCAMPlanner(G, seed=0).plan(starts, goals)
        _assert_solution_valid(self, G, result, starts, goals)

    def test_deterministic_given_seed(self):
        G = _grid(3, 3)
        starts = {"a": "0,0", "b": "2,2"}
        goals = {"a": "2,2", "b": "0,0"}
        r1 = LaCAMPlanner(G, seed=5).plan(starts, goals)
        r2 = LaCAMPlanner(G, seed=5).plan(starts, goals)
        self.assertEqual(r1.paths, r2.paths)

    def test_already_at_goals(self):
        G = _line(3)
        starts = {"a": "0", "b": "2"}
        result = LaCAMPlanner(G, seed=0).plan(starts, dict(starts))
        self.assertTrue(result.success)
        self.assertEqual(result.makespan, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
