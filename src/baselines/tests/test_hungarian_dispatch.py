"""Tests for the Sprint 5 Hungarian (LSAP) dispatcher."""

from __future__ import annotations

import math
import unittest

import networkx as nx

from src.baselines.hungarian_dispatch import hungarian_dispatch
from src.env.warehouse_env import AgentState, DispatchContext, WarehouseEnv
from src.routing.astar import AStarRouter


def _ctx(idle, tasks, dist):
    return DispatchContext(
        idle_agents=tuple(idle), pending_tasks=tuple(tasks), distance=dist
    )


class TestHungarianDispatch(unittest.TestCase):
    def test_optimal_beats_greedy(self):
        # Greedy (A grabs nearest p1) forces B onto far p2: 1+10=11.
        # LSAP global optimum is A->p2, B->p1: 2+1=3.
        D = {("a", "p1"): 1.0, ("a", "p2"): 2.0, ("b", "p1"): 1.0, ("b", "p2"): 10.0}
        ctx = _ctx(
            [("A", "a", 0.0), ("B", "b", 0.0)],
            [(1, "p1", "x"), (2, "p2", "x")],
            lambda s, t: D[(s, t)],
        )
        self.assertEqual(hungarian_dispatch(ctx), {"A": 2, "B": 1})

    def test_empty_inputs_return_empty(self):
        d = lambda s, t: 1.0  # noqa: E731
        self.assertEqual(hungarian_dispatch(_ctx([], [(1, "p", "x")], d)), {})
        self.assertEqual(hungarian_dispatch(_ctx([("A", "a", 0.0)], [], d)), {})

    def test_unreachable_pickup_skipped(self):
        D = {("a", "p1"): math.inf, ("a", "p2"): 5.0}
        ctx = _ctx(
            [("A", "a", 0.0)], [(1, "p1", "x"), (2, "p2", "x")],
            lambda s, t: D[(s, t)],
        )
        self.assertEqual(hungarian_dispatch(ctx), {"A": 2})  # avoids unreachable p1

    def test_all_unreachable_returns_empty(self):
        ctx = _ctx([("A", "a", 0.0)], [(1, "p1", "x")], lambda s, t: math.inf)
        self.assertEqual(hungarian_dispatch(ctx), {})

    def test_rectangular_more_agents_than_tasks(self):
        # 3 agents, 2 tasks -> exactly 2 assigned, one-to-one, cheapest pairs.
        D = {
            ("a", "p"): 1.0, ("b", "p"): 5.0, ("c", "p"): 9.0,
            ("a", "q"): 2.0, ("b", "q"): 1.0, ("c", "q"): 9.0,
        }
        ctx = _ctx(
            [("A", "a", 0.0), ("B", "b", 0.0), ("C", "c", 0.0)],
            [(1, "p", "x"), (2, "q", "x")],
            lambda s, t: D[(s, t)],
        )
        out = hungarian_dispatch(ctx)
        self.assertEqual(out, {"A": 1, "B": 2})  # C unassigned, total cost 2


class TestHungarianDispatchIntegration(unittest.TestCase):
    """Plugs into the real env via the Sprint-5 dispatch hook."""

    @staticmethod
    def _env():
        G = nx.DiGraph()
        for x in range(5):
            G.add_node(str(x), x=float(x), y=0.0)
        for x in range(4):
            G.add_edge(str(x), str(x + 1), length=1.0)
            G.add_edge(str(x + 1), str(x), length=1.0)
        router = AStarRouter(G, precompute=True)
        return WarehouseEnv(
            graph=G, router=router, num_agents=2, episode_horizon=8,
            task_rate=0.0, knn_agents=1, seed=0,
        )

    def test_dispatcher_assigns_through_hook(self):
        env = self._env()
        env.set_dispatcher(hungarian_dispatch)
        env.reset(seed=0)
        assigned = [i for i in env._agent_info.values() if i.task is not None]
        self.assertTrue(assigned, "LSAP assigned idle agents from preseeded pool")
        for info in assigned:
            self.assertEqual(info.state, AgentState.TO_PICKUP)
        ids = [i.task.id for i in assigned]
        self.assertEqual(len(ids), len(set(ids)), "one-to-one task assignment")


if __name__ == "__main__":
    unittest.main(verbosity=2)
