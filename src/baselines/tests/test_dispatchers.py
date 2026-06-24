"""Tests for FIFO-nearest and Hungarian assignment baselines."""

from __future__ import annotations

import unittest

import networkx as nx

from src.baselines.dispatchers import (
    fifo_nearest_goal_assignment,
    fifo_nearest_task_assignment,
    hungarian_goal_assignment,
    hungarian_task_assignment,
)
from src.env.task_generator import Task
from src.routing.astar import AStarRouter


def _line_graph(n: int) -> nx.DiGraph:
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(str(i), x=float(i), y=0.0, is_halt=False, type="POINT")
    for i in range(n - 1):
        G.add_edge(str(i), str(i + 1), weight=1.0, length=1.0)
        G.add_edge(str(i + 1), str(i), weight=1.0, length=1.0)
    return G


class TestDispatchers(unittest.TestCase):
    def setUp(self):
        self.G = _line_graph(5)
        self.router = AStarRouter(self.G, precompute=True)
        self.agent_positions = {"a0": "0", "a1": "4"}

    def test_fifo_nearest_tasks_keep_task_order(self):
        tasks = [
            Task(id=0, pickup="3", dropoff="0", spawn_step=0),
            Task(id=1, pickup="1", dropoff="4", spawn_step=0),
        ]
        assignments = fifo_nearest_task_assignment(
            self.agent_positions,
            tasks,
            self.router,
        )
        self.assertEqual(assignments["a1"].task.id, 0)
        self.assertEqual(assignments["a0"].task.id, 1)

    def test_hungarian_tasks_minimize_global_pickup_distance(self):
        tasks = [
            Task(id=0, pickup="1", dropoff="4", spawn_step=0),
            Task(id=1, pickup="3", dropoff="0", spawn_step=0),
        ]
        assignments = hungarian_task_assignment(
            self.agent_positions,
            tasks,
            self.router,
        )
        self.assertEqual(assignments["a0"].task.id, 0)
        self.assertEqual(assignments["a1"].task.id, 1)

    def test_goal_assignment_variants_return_goal_maps(self):
        goals = ["3", "1"]
        fifo = fifo_nearest_goal_assignment(self.agent_positions, goals, self.router)
        hungarian = hungarian_goal_assignment(self.agent_positions, goals, self.router)
        self.assertEqual(fifo["a1"].goal, "3")
        self.assertEqual(fifo["a0"].goal, "1")
        self.assertEqual(hungarian["a0"].goal, "1")
        self.assertEqual(hungarian["a1"].goal, "3")


if __name__ == "__main__":
    unittest.main(verbosity=2)
