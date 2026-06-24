"""Tests for the Sprint 4b P0 LaCAM lifelong reference."""

from __future__ import annotations

import unittest

import networkx as nx

from src.baselines.lacam_lifelong import (
    LaCAMLifelongController,
    run_episode,
    summarize,
)
from src.env.warehouse_env import WarehouseEnv
from src.mapf.lacam import LaCAMPlanner
from src.routing.astar import AStarRouter


def _line_env(num_agents=2, horizon=64, n_nodes=4, task_budget=None):
    G = nx.DiGraph()
    for x in range(n_nodes):
        G.add_node(str(x), x=float(x), y=0.0)
    for x in range(n_nodes - 1):
        G.add_edge(str(x), str(x + 1), weight=1.0, length=1.0)
        G.add_edge(str(x + 1), str(x), weight=1.0, length=1.0)
    router = AStarRouter(G, precompute=True)
    return WarehouseEnv(
        graph=G,
        router=router,
        num_agents=num_agents,
        episode_horizon=horizon,
        task_rate=0.0,
        knn_agents=1,
        task_budget=task_budget,
        seed=0,
    )


class TestLaCAMLifelong(unittest.TestCase):
    def test_controller_uses_lacam_planner(self):
        env = _line_env()
        env.reset(seed=0)
        ctrl = LaCAMLifelongController(env, seed=0)
        self.assertIsInstance(ctrl.planner, LaCAMPlanner)

    def test_single_agent_completes_tasks(self):
        env = _line_env(num_agents=1, horizon=32, n_nodes=4)
        row = run_episode(env, seed=0)
        self.assertGreaterEqual(int(row["tasks_completed"]), 1)
        self.assertEqual(row["baseline"], "lacam_lifelong")
        self.assertEqual(row["steps"], 32)

    def test_row_schema(self):
        env = _line_env(num_agents=2, horizon=24)
        row = run_episode(env, seed=0)
        for key in (
            "baseline", "seed", "steps", "tasks_completed",
            "task_budget_reached", "plan_failures", "freeze_sidesteps", "wall_s",
        ):
            self.assertIn(key, row)

    def test_reactive_fallback_greedy_window_steps_toward_goal(self):
        """R2/DA fallback: a failed window must degrade to greedy movement, not a
        fleet-wide freeze. _greedy_window returns the next nodes toward goal."""
        env = _line_env(num_agents=2, horizon=24, n_nodes=5)
        env.reset(seed=0)
        ctrl = LaCAMLifelongController(env, seed=0, reactive_fallback=True)
        q = ctrl._greedy_window({"x": "0"}, {"x": "4"})
        self.assertTrue(q["x"], "fallback must move, not freeze")
        self.assertEqual(q["x"][0], "1")  # first step is toward the goal
        # an agent already at its goal holds (empty queue, no spurious move)
        self.assertEqual(ctrl._greedy_window({"y": "4"}, {"y": "4"})["y"], [])

    def test_task_budget_success_regime(self):
        env = _line_env(num_agents=1, horizon=64, n_nodes=4, task_budget=1)
        row = run_episode(env, seed=0)
        self.assertEqual(row["task_budget_reached"], 1)
        self.assertLess(int(row["steps"]), 64)

    def test_summarize_success_block(self):
        rows = [
            {"tasks_completed": 11, "plan_failures": 0, "replans": 5,
             "task_budget_reached": 1, "steps": 300},
            {"tasks_completed": 4, "plan_failures": 1, "replans": 6,
             "task_budget_reached": 0, "steps": 4096},
        ]
        s = summarize(rows, task_budget=11)
        self.assertEqual(s["baseline"], "lacam_lifelong")
        self.assertEqual(s["success_count"], 1)
        self.assertEqual(s["success_rate"], 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
