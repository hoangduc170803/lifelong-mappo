"""Tests for Sprint 2 benchmark CSV runner."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import networkx as nx

from src.baselines.benchmark import (
    run_benchmark,
    run_mapf_baseline,
    sample_mapf_instance,
    benchmark_row,
    write_benchmark_csv,
)
from src.routing.astar import AStarRouter


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


class TestBenchmarkRunner(unittest.TestCase):
    def test_sample_mapf_instance_is_deterministic_and_distinct(self):
        G = _grid_graph(4, 4)
        starts_a, goals_a = sample_mapf_instance(G, num_agents=3, seed=42)
        starts_b, goals_b = sample_mapf_instance(G, num_agents=3, seed=42)
        self.assertEqual(starts_a, starts_b)
        self.assertEqual(goals_a, goals_b)
        self.assertEqual(len(set(starts_a.values()) | set(goals_a.values())), 6)

    def test_runs_priority_search_and_hungarian_prioritized_rows(self):
        G = _grid_graph(4, 4)
        router = AStarRouter(G, precompute=True)
        starts = {"a0": "0,0", "a1": "3,3"}
        goals = {"a0": "3,3", "a1": "0,0"}

        priority_run = run_mapf_baseline(
            G,
            starts,
            goals,
            baseline="priority_search",
            max_time=16,
            use_external=False,
            router=router,
        )
        self.assertTrue(priority_run.result.success, priority_run.result.diagnostics)
        priority_row = benchmark_row(
            "priority_search",
            0,
            2,
            starts,
            priority_run.goals,
            priority_run,
            router=router,
        )
        self.assertEqual(priority_row.success_rate, 1.0)
        self.assertEqual(priority_row.conflicts_total, 0)
        self.assertGreater(priority_row.lower_bound_steps, 0)
        self.assertGreaterEqual(priority_row.elapsed_s, priority_row.elapsed_planner_s)

        cbs_run = run_mapf_baseline(
            G,
            starts,
            goals,
            baseline="hungarian_prioritized_planning",
            max_time=16,
            use_external=False,
            router=router,
        )
        self.assertTrue(cbs_run.result.success, cbs_run.result.diagnostics)
        self.assertEqual(set(cbs_run.goals), set(starts))
        self.assertGreater(cbs_run.elapsed_assignment_s, 0.0)

        opentcs_run = run_mapf_baseline(
            G,
            starts,
            goals,
            baseline="opentcs_naive",
            max_time=16,
            use_external=False,
            router=router,
        )
        self.assertEqual(opentcs_run.result.solver, "opentcs_naive")
        self.assertEqual(set(opentcs_run.goals), set(starts))
        self.assertGreaterEqual(opentcs_run.elapsed_assignment_s, 0.0)

    def test_run_benchmark_writes_csv(self):
        G = _grid_graph(4, 4)
        rows = run_benchmark(
            G,
            agent_counts=[2],
            seeds=[0],
            baselines=[
                "priority_search",
                "fifo_nearest_prioritized_planning",
                "prioritized_planning_default_order",
            ],
            max_time=16,
            use_external=False,
        )
        self.assertEqual(len(rows), 3)
        with tempfile.TemporaryDirectory() as tmp:
            out = write_benchmark_csv(rows, Path(tmp) / "baselines.csv")
            self.assertTrue(out.exists())
            text = out.read_text(encoding="utf-8")
            self.assertIn("instance_makespan", text)
            self.assertIn("elapsed_assignment_s", text)
            self.assertIn("lower_bound_steps", text)
            self.assertIn("failure_reason", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
