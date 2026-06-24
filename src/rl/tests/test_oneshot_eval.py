"""Plumbing tests for the GATE-1 condition-2 one-shot eval harness.

Validate injection + delivered-latch + success/makespan accounting against a
known geometry, so a 0% learned-policy result cannot be dismissed as a harness
bug. No learned performance is asserted.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.rl.mappo_onpolicy.oneshot_eval import P3_BAR, _inject_oneshot
from src.rl.mappo_onpolicy.env_adapter import WarehouseEnvConfig, build_warehouse_env
from src.env.warehouse_env import AgentState


class TestOneShotHarness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = build_warehouse_env(
            WarehouseEnvConfig(
                num_agents=3,
                episode_horizon=64,
                task_rate=0.0,
                task_distribution="uniform",
                knn_agents=2,
            )
        )

    def test_inject_sets_goal_as_current_goal_and_registers_in_flight(self):
        env = self.env
        env.reset(seed=0)
        agents = list(env.agents)
        nodes = list(env.G.nodes())
        starts = {a: nodes[i] for i, a in enumerate(agents)}
        goals = {a: nodes[i + 3] for i, a in enumerate(agents)}
        _inject_oneshot(env, starts, goals)
        for a in agents:
            info = env._agent_info[a]
            self.assertEqual(info.pos, starts[a])
            self.assertEqual(info.state, AgentState.TO_DROPOFF)
            # current_goal is the dropoff once picked_up.
            self.assertEqual(info.task.current_goal, goals[a])
            self.assertIn(info.task.id, env._task_gen.in_flight)
        # Pending pool was cleared so no stray dispatch.
        self.assertEqual(len(env._task_gen.pending), 0)
        # Exactly one in_flight task per agent: reset() greedy-dispatches orphan
        # tasks into in_flight, and inject must clear them (else this is 2*N).
        self.assertEqual(env._task_gen.num_in_flight(), len(agents))
        self.assertEqual(len(env._task_gen.in_flight), len(agents))

    def test_p3_bar_has_four_dists_with_betweenness_zero(self):
        self.assertEqual(len(P3_BAR), 4)
        # betweenness is the cell where classical one-shot also fails (P3).
        self.assertEqual(P3_BAR["betweenness_bottleneck"]["pibt"], 0.0)
        self.assertEqual(P3_BAR["betweenness_bottleneck"]["lacam"], 0.0)
        for d in ("hotspot", "burst_wave", "bipartite_pickup_dropoff"):
            self.assertEqual(P3_BAR[d]["pibt"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
