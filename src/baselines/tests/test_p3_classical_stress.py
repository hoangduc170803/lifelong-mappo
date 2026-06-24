"""Tests for the Sprint 4b P3 one-shot classical stress runner."""

from __future__ import annotations

import unittest

import networkx as nx

from src.baselines.p3_classical_stress import (
    run_lacam_oneshot,
    run_pibt_oneshot,
    summarize,
)
from src.routing.astar import AStarRouter


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
                    for u, v in (
                        (f"{x},{y}", f"{nx_},{ny_}"),
                        (f"{nx_},{ny_}", f"{x},{y}"),
                    ):
                        G.add_edge(u, v, weight=1.0, length=1.0)
    return G


class TestP3OneShot(unittest.TestCase):
    def setUp(self):
        self.G = _grid(4, 4)
        self.router = AStarRouter(self.G, precompute=True)

    def test_pibt_oneshot_reaches_crossing_goals(self):
        starts = {"a": "0,0", "b": "3,3"}
        goals = {"a": "3,3", "b": "0,0"}
        row = run_pibt_oneshot(self.G, self.router.distance, starts, goals, seed=0)
        self.assertTrue(row["success"])
        self.assertEqual(row["conflicts"], 0)
        self.assertGreater(row["makespan"], 0)
        self.assertEqual(row["baseline"], "pibt_oneshot")

    def test_lacam_oneshot_reaches_crossing_goals(self):
        starts = {"a": "0,0", "b": "3,3", "c": "0,3", "d": "3,0"}
        goals = {"a": "3,3", "b": "0,0", "c": "3,0", "d": "0,3"}
        row = run_lacam_oneshot(self.G, self.router.distance, starts, goals, seed=0)
        self.assertTrue(row["success"])
        self.assertEqual(row["conflicts"], 0)
        self.assertEqual(row["baseline"], "lacam_oneshot")

    def test_pibt_already_at_goal_is_zero_makespan_success(self):
        starts = {"a": "1,1", "b": "2,2"}
        row = run_pibt_oneshot(self.G, self.router.distance, starts, dict(starts), seed=0)
        self.assertTrue(row["success"])
        self.assertEqual(row["makespan"], 0)

    def test_pibt_step_cap_reports_failure_not_hang(self):
        # A 2-node line: a head-on swap is physically unsolvable; PIBT must
        # exhaust the cap and report failure rather than loop forever.
        G = nx.DiGraph()
        G.add_node("0", x=0.0, y=0.0)
        G.add_node("1", x=1.0, y=0.0)
        G.add_edge("0", "1", weight=1.0, length=1.0)
        G.add_edge("1", "0", weight=1.0, length=1.0)
        router = AStarRouter(G, precompute=True)
        row = run_pibt_oneshot(
            G, router.distance, {"a": "0", "b": "1"}, {"a": "1", "b": "0"},
            seed=0, step_cap=20,
        )
        self.assertFalse(row["success"])

    def test_summarize_groups_by_baseline_and_distribution(self):
        rows = [
            {"baseline": "pibt_oneshot", "distribution": "hotspot", "success": True, "makespan": 10, "elapsed_s": 0.1},
            {"baseline": "pibt_oneshot", "distribution": "hotspot", "success": False, "makespan": 0, "elapsed_s": 0.2},
            {"baseline": "lacam_oneshot", "distribution": "hotspot", "success": True, "makespan": 12, "elapsed_s": 0.05},
        ]
        summary = summarize(rows)
        cells = {(c["baseline"], c["distribution"]): c for c in summary["cells"]}
        pibt = cells[("pibt_oneshot", "hotspot")]
        self.assertEqual(pibt["success_count"], 1)
        self.assertEqual(pibt["success_rate"], 0.5)
        self.assertEqual(pibt["makespan_mean"], 10)  # only successes counted
        lacam = cells[("lacam_oneshot", "hotspot")]
        self.assertEqual(lacam["success_count"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
