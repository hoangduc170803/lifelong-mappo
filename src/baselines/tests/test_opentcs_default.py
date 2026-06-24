"""Tests for the OpenTCS default-stack emulator baseline."""

from __future__ import annotations

import unittest

import networkx as nx

from src.baselines.opentcs_default import (
    OpenTCSNaiveEmulator,
    opentcs_default_goal_assignment,
)
from src.mapf.cbs_solver import validate_paths
from src.routing.astar import AStarRouter


def _line_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    for idx in range(4):
        G.add_node(str(idx), x=float(idx), y=0.0, is_halt=False, type="POINT")
    for idx in range(3):
        G.add_edge(str(idx), str(idx + 1), weight=1.0, length=1.0)
        G.add_edge(str(idx + 1), str(idx), weight=1.0, length=1.0)
    return G


def _merge_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    nodes = {
        "a_start": (0.0, 0.0),
        "b_start": (0.0, 1.0),
        "center": (1.0, 0.5),
        "a_goal": (2.0, 0.0),
        "b_goal": (2.0, 1.0),
    }
    for node, (x, y) in nodes.items():
        G.add_node(node, x=x, y=y, is_halt=False, type="POINT")
    for src, dst in [
        ("a_start", "center"),
        ("b_start", "center"),
        ("center", "a_goal"),
        ("center", "b_goal"),
    ]:
        G.add_edge(src, dst, weight=1.0, length=1.0)
    return G


class TestOpenTCSNaiveEmulator(unittest.TestCase):
    def test_vehicle_first_cost_greedy_assignment(self):
        G = _line_graph()
        router = AStarRouter(G, precompute=True)
        assignments = opentcs_default_goal_assignment(
            {"agv_0": "0", "agv_1": "3"},
            ["2", "1"],
            router,
        )
        self.assertEqual(assignments["agv_0"].goal, "1")
        self.assertEqual(assignments["agv_1"].goal, "2")

    def test_scheduler_emulator_serializes_shared_center(self):
        G = _merge_graph()
        router = AStarRouter(G, precompute=True)
        planner = OpenTCSNaiveEmulator(G, max_time=8)
        result = planner.plan(
            starts={"agv_0": "a_start", "agv_1": "b_start"},
            goals={"agv_0": "a_goal", "agv_1": "b_goal"},
            router=router,
        )
        self.assertTrue(result.success, result.diagnostics)
        self.assertEqual(validate_paths(G, result.paths), [])
        self.assertGreater(result.diagnostics["blocked_move_requests"], 0)
        self.assertEqual(result.solver, "opentcs_naive")

    def test_head_on_swap_times_out_without_deadlock_recovery(self):
        G = _line_graph()
        router = AStarRouter(G, precompute=True)
        planner = OpenTCSNaiveEmulator(G, max_time=8)
        result = planner.plan(
            starts={"agv_0": "0", "agv_1": "3"},
            goals={"agv_0": "3", "agv_1": "0"},
            router=router,
        )
        self.assertFalse(result.success)
        self.assertTrue(result.diagnostics["timed_out"])
        self.assertEqual(result.diagnostics["deadlock_recovery"], "not_emulated")


if __name__ == "__main__":
    unittest.main(verbosity=2)
