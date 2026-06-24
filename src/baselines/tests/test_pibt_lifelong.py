"""Tests for the Sprint 4a D6 PIBT lifelong reference runner."""

from __future__ import annotations

import unittest

import networkx as nx

from src.baselines.pibt_lifelong import (
    PIBTLifelongController,
    run_episode,
    summarize,
)
from src.env.compass_mapper import WAIT_SLOT
from src.env.warehouse_env import WarehouseEnv
from src.routing.astar import AStarRouter


def _line_env(num_agents=2, horizon=64, n_nodes=4, task_budget=None):
    G = nx.DiGraph()
    for x in range(n_nodes):
        G.add_node(str(x), x=float(x), y=0.0)
    for x in range(n_nodes - 1):
        G.add_edge(str(x), str(x + 1), length=1.0)
        G.add_edge(str(x + 1), str(x), length=1.0)
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


class TestPIBTLifelong(unittest.TestCase):
    def test_actions_are_mask_valid(self):
        """Every issued action must be valid under the env's own mask."""
        env = _line_env(num_agents=2, horizon=16)
        env.reset(seed=0)
        ctrl = PIBTLifelongController(env, seed=0)
        for _ in range(16):
            masks, _, _ = env._action_masks_for_agents(env.agents)
            acts = ctrl.actions()
            for a, slot in acts.items():
                self.assertTrue(
                    masks[a][slot],
                    f"agent {a} issued mask-invalid slot {slot}",
                )
            _, _, term, trunc, _ = env.step(acts)
            if all(trunc.values()) or all(term.values()):
                break

    def test_single_agent_completes_tasks(self):
        env = _line_env(num_agents=1, horizon=32)
        row = run_episode(env, seed=0)
        self.assertGreaterEqual(int(row["tasks_completed"]), 1)
        self.assertEqual(row["steps"], 32)
        self.assertEqual(row["baseline"], "pibt_lifelong")

    def test_two_agents_row_schema_and_progress(self):
        env = _line_env(num_agents=2, horizon=48)
        row = run_episode(env, seed=0)
        for key in (
            "baseline",
            "seed",
            "steps",
            "tasks_completed",
            "throughput_per_step",
            "validator_interventions",
            "wall_s",
        ):
            self.assertIn(key, row)

    def test_priorities_grow_until_goal_then_reset(self):
        env = _line_env(num_agents=1, horizon=16)
        env.reset(seed=0)
        ctrl = PIBTLifelongController(env, seed=0)
        agent = env.agents[0]
        # Pin a task whose pickup provably differs from the agent's position
        # (greedy dispatch may otherwise assign a pickup at the agent's node).
        info = env._agent_info[agent]
        info.pos = info.prev_pos = "0"
        task = env._task_gen.create_task(step_idx=0)
        task.pickup, task.dropoff = "3", "0"
        info.task = task
        ctrl.actions()
        first = ctrl._priorities[agent]
        ctrl.actions()
        second = ctrl._priorities[agent]
        # Agent has a task it has not reached -> priority strictly grows.
        self.assertGreater(second, first)
        # On (simulated) arrival the priority resets.
        info.pos = "3"
        ctrl.actions()
        # current_goal is still the pickup until env marks it picked up, so
        # pos == goal now -> reset branch.
        self.assertEqual(ctrl._priorities[agent], 0.0)

    def test_freeze_sidestep_dissolves_static_blob(self):
        """REGRESSION: PIBT lifelong must not freeze forever in a blob.

        First 4096-step N=20 run: every seed completed its last task by
        step ~60-1000 and then nothing moved for thousands of ticks —
        boundary agents had an open exit but preferred staying (closer to
        goal). After `freeze_patience` static ticks one stuck agent must be
        forced onto an open slot."""
        env = _line_env(num_agents=2, horizon=64)
        env.reset(seed=0)
        a0, a1 = sorted(env.agents)
        # Head-on geometry on the 0-1-2-3 line: a0 at "1" wants "3", a1 at
        # "2" wants "0". Each one's best move is the other's node, so plain
        # masked PIBT leaves both staying forever; a0's open exit is "0",
        # a1's is "3".
        i0, i1 = env._agent_info[a0], env._agent_info[a1]
        t0 = env._task_gen.create_task(step_idx=0)
        t0.pickup, t0.dropoff = "3", "0"
        t1 = env._task_gen.create_task(step_idx=0)
        t1.pickup, t1.dropoff = "0", "3"
        i0.pos = i0.prev_pos = "1"
        i0.task = t0
        i1.pos = i1.prev_pos = "2"
        i1.task = t1

        ctrl = PIBTLifelongController(env, seed=0)
        sidestepped = False
        for _ in range(ctrl.freeze_patience + 3):
            acts = ctrl.actions()
            if ctrl.freeze_sidesteps:
                non_wait = [a for a, s in acts.items() if s != WAIT_SLOT]
                self.assertGreaterEqual(len(non_wait), 1)
                sidestepped = True
                break
            # Both agents mutually blocked -> everyone waits, positions
            # unchanged (we do NOT env.step, so the streak grows).
        self.assertTrue(sidestepped)

    def test_summarize_shape(self):
        rows = [{"tasks_completed": 40}, {"tasks_completed": 60}]
        summary = summarize(rows)
        self.assertEqual(summary["tasks_completed_mean"], 50.0)
        self.assertEqual(summary["d4_criterion_0.9x"], 45.0)
        self.assertNotIn("success_rate", summary)

    def test_task_budget_terminates_episode_early_with_success(self):
        env = _line_env(num_agents=1, horizon=64, task_budget=1)
        row = run_episode(env, seed=0)
        self.assertEqual(row["task_budget_reached"], 1)
        self.assertGreaterEqual(int(row["tasks_completed"]), 1)
        self.assertLess(int(row["steps"]), 64)

    def test_summarize_success_regime_block(self):
        rows = [
            {"tasks_completed": 200, "task_budget_reached": 1, "steps": 600},
            {"tasks_completed": 200, "task_budget_reached": 1, "steps": 800},
            {"tasks_completed": 120, "task_budget_reached": 0, "steps": 1024},
        ]
        summary = summarize(rows, task_budget=200)
        self.assertEqual(summary["success_count"], 2)
        self.assertEqual(summary["success_rate"], 0.667)
        self.assertEqual(summary["steps_to_budget_mean"], 700)
        self.assertEqual(summary["steps_to_budget_max"], 800)


if __name__ == "__main__":
    unittest.main(verbosity=2)
