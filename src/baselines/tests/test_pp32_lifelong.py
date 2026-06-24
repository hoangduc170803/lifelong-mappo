"""Tests for the Sprint 4a D5 PP-32 lifelong reference runner."""

from __future__ import annotations

import unittest

import networkx as nx

from src.baselines.pp32_lifelong import (
    PP32LifelongController,
    run_episode,
    summarize,
)
from src.env.compass_mapper import WAIT_SLOT
from src.env.warehouse_env import WarehouseEnv
from src.mapf.cbs_solver import MAPFPlanResult
from src.routing.astar import AStarRouter


def _line_graph(n=3):
    G = nx.DiGraph()
    for x in range(n):
        G.add_node(str(x), x=float(x), y=0.0)
    for x in range(n - 1):
        G.add_edge(str(x), str(x + 1), length=1.0)
        G.add_edge(str(x + 1), str(x), length=1.0)
    return G


def _line_env(num_agents=2, horizon=64, n_nodes=3, task_budget=None):
    G = _line_graph(n_nodes)
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


class TestPP32LifelongController(unittest.TestCase):
    def test_slot_mapping_neighbor_and_non_neighbor(self):
        env = _line_env()
        env.reset(seed=0)
        ctrl = PP32LifelongController(env, seed=0)
        slot = ctrl._slot_to("0", "1")
        self.assertIsNotNone(slot)
        self.assertNotEqual(slot, WAIT_SLOT)
        # "2" is not a compass neighbor of "0" on a 3-node line.
        self.assertIsNone(ctrl._slot_to("0", "2"))

    def test_planned_wait_is_consumed_and_next_node_mapped(self):
        env = _line_env(num_agents=1, n_nodes=4)
        env.reset(seed=0)
        agent = env.agents[0]
        info = env._agent_info[agent]
        info.pos = "1"
        ctrl = PP32LifelongController(env, seed=0)
        # Pretend a plan exists: wait in place once, then move to "2".
        ctrl._steps_since_plan = 0
        ctrl._planned_goals = ctrl._current_goals()
        ctrl._queues = {agent: ["1", "2"]}
        # Force the goal map to match so no replan triggers.
        ctrl._planned_goals = ctrl._current_goals()

        acts = ctrl.actions()
        # In-place wait entries are consumed and the next hop is mapped.
        self.assertEqual(ctrl._queues[agent], ["2"])
        self.assertEqual(acts[agent], ctrl._slot_to("1", "2"))

    def test_plan_failure_means_all_wait(self):
        env = _line_env(num_agents=2)
        env.reset(seed=0)
        ctrl = PP32LifelongController(env, seed=0)
        ctrl.planner.plan = lambda *a, **k: MAPFPlanResult(  # type: ignore[assignment]
            success=False, solver="priority_search"
        )
        acts = ctrl.actions()
        self.assertEqual(ctrl.plan_failures, 1)
        for action in acts.values():
            self.assertEqual(action, WAIT_SLOT)

    def test_plan_failure_backs_off_until_horizon(self):
        """REGRESSION: a failed replan must NOT retry every tick.

        First reference run livelocked on seed 0 (155 failures, ~6000 s)
        because empty post-failure queues re-triggered the "queue exhausted"
        replan condition each tick. After a failure the controller must wait
        for a goal change or the periodic horizon.
        """
        env = _line_env(num_agents=2, horizon=64)
        env.reset(seed=0)
        ctrl = PP32LifelongController(env, replan_horizon=20, seed=0)
        ctrl.planner.plan = lambda *a, **k: MAPFPlanResult(  # type: ignore[assignment]
            success=False, solver="priority_search"
        )

        # Tick 1 plans (and fails); ticks 2..20 must not replan (goals are
        # unchanged because everyone waits in place after a failure).
        for _ in range(ctrl.replan_horizon):
            ctrl.actions()
        self.assertEqual(ctrl.plan_failures, 1)
        # Tick 21 crosses the periodic horizon: exactly one retry.
        ctrl.actions()
        self.assertEqual(ctrl.plan_failures, 2)

    def test_plan_failure_retries_on_goal_change(self):
        env = _line_env(num_agents=1, n_nodes=4)
        env.reset(seed=0)
        agent = env.agents[0]
        ctrl = PP32LifelongController(env, replan_horizon=20, seed=0)
        ctrl.planner.plan = lambda *a, **k: MAPFPlanResult(  # type: ignore[assignment]
            success=False, solver="priority_search"
        )
        ctrl.actions()
        self.assertEqual(ctrl.plan_failures, 1)
        ctrl.actions()
        self.assertEqual(ctrl.plan_failures, 1)  # backed off
        # Simulate dispatch flipping the agent's goal -> immediate retry.
        # (Reset pre-seeds the pool, so the agent already holds a task; pick a
        # pickup that provably differs from both its position and old goal.)
        info = env._agent_info[agent]
        info.pos = "0"
        old_goal = ctrl._planned_goals[agent]
        new_pickup = next(n for n in ("3", "2", "1") if n not in ("0", old_goal))
        task = env._task_gen.create_task(step_idx=0)
        task.pickup, task.dropoff = new_pickup, "0"
        info.task = task
        ctrl.actions()
        self.assertEqual(ctrl.plan_failures, 2)

    def test_forced_wait_does_not_advance_queue(self):
        env = _line_env(num_agents=1, n_nodes=4)
        env.reset(seed=0)
        agent = env.agents[0]
        env._agent_info[agent].pos = "1"
        ctrl = PP32LifelongController(env, seed=0)
        ctrl._queues = {agent: ["2", "3"]}
        # Agent did NOT reach "2" (e.g. validator forced a wait).
        ctrl.observe_transition()
        self.assertEqual(ctrl._queues[agent], ["2", "3"])
        # Agent did reach "2": head is consumed.
        env._agent_info[agent].pos = "2"
        ctrl.observe_transition()
        self.assertEqual(ctrl._queues[agent], ["3"])

    def test_single_agent_completes_tasks_and_replans_on_goal_change(self):
        env = _line_env(num_agents=1, horizon=32, n_nodes=4)
        row = run_episode(env, seed=0)
        # A single agent on a 4-node line with pre-seeded tasks must complete
        # at least one pickup->dropoff cycle in 32 steps.
        self.assertGreaterEqual(int(row["tasks_completed"]), 1)
        # Goal flips (pickup reached, task completed) force event replans on
        # top of the initial plan.
        self.assertGreaterEqual(int(row["replans"]), 2)
        self.assertEqual(row["plan_failures"], 0)
        self.assertEqual(row["steps"], 32)

    def test_freeze_sidesteps_one_blocked_agent(self):
        """REGRESSION: symmetric lookahead-mask contention must be broken.

        Two agents whose A* hints claim the same free node block each other
        forever (seen on seeds 2/5/6/7 of the first reference run: 6-10
        tasks over 1024 steps, convoys frozen behind one mutual claim).
        After `freeze_patience` static ticks, exactly one blocked agent must
        sidestep to an unmasked neighbor and drop its queue."""
        env = _line_env(num_agents=2, horizon=64, n_nodes=4)
        env.reset(seed=0)
        a0, a1 = env.agents
        # Force the mutual-claim geometry on the 0-1-2-3 line: a0 at "1"
        # heading to "3", a1 at "3" heading to "0" — both A* hints claim "2".
        info0, info1 = env._agent_info[a0], env._agent_info[a1]
        t0 = env._task_gen.create_task(step_idx=0)
        t0.pickup, t0.dropoff = "3", "0"
        t1 = env._task_gen.create_task(step_idx=0)
        t1.pickup, t1.dropoff = "0", "3"
        info0.pos = info0.prev_pos = "1"
        info0.task = t0
        info1.pos = info1.prev_pos = "3"
        info1.task = t1

        ctrl = PP32LifelongController(env, replan_horizon=50, seed=0)
        ctrl._planned_goals = ctrl._current_goals()
        ctrl._queues = {a0: ["2", "3"], a1: ["2", "1", "0"]}
        ctrl._steps_since_plan = 0

        # Sanity: the mask must block "2" for both (mutual prediction claim).
        masks, _, _ = env._action_masks_for_agents(env.agents)
        self.assertFalse(masks[a0][ctrl._slot_to("1", "2")])

        sidestepped = False
        for _ in range(ctrl.freeze_patience + 2):
            acts = ctrl.actions()
            if ctrl.freeze_sidesteps:
                # Exactly one agent got a non-WAIT action despite the block.
                non_wait = [a for a, s in acts.items() if s != WAIT_SLOT]
                self.assertEqual(len(non_wait), 1)
                self.assertEqual(ctrl._queues[non_wait[0]], [])
                sidestepped = True
                break
            for action in acts.values():
                self.assertEqual(action, WAIT_SLOT)
            ctrl.observe_transition()  # positions unchanged -> static++
        self.assertTrue(sidestepped)

    def test_two_agents_full_episode_row_schema(self):
        env = _line_env(num_agents=2, horizon=24)
        row = run_episode(env, seed=0)
        for key in (
            "baseline",
            "seed",
            "steps",
            "tasks_completed",
            "throughput_per_step",
            "replans",
            "plan_failures",
            "freeze_sidesteps",
            "orders_tried_total",
            "plan_time_s",
            "validator_interventions",
            "wall_s",
        ):
            self.assertIn(key, row)
        self.assertEqual(row["baseline"], "pp32_lifelong")
        self.assertEqual(row["steps"], 24)

    def test_summarize_reports_mean_and_criterion(self):
        rows = [
            {"tasks_completed": 50, "plan_failures": 0, "replans": 10},
            {"tasks_completed": 60, "plan_failures": 1, "replans": 12},
        ]
        summary = summarize(rows)
        self.assertEqual(summary["tasks_completed_mean"], 55.0)
        self.assertEqual(summary["d4_criterion_0.9x"], 49.5)
        self.assertEqual(summary["plan_failures_total"], 1)
        self.assertNotIn("success_rate", summary)

    def test_task_budget_terminates_episode_early_with_success(self):
        """Success regime: env's task_budget terminal ends the episode
        before the step cap and the row records the success."""
        env = _line_env(num_agents=1, horizon=64, n_nodes=4, task_budget=1)
        row = run_episode(env, seed=0)
        self.assertEqual(row["task_budget_reached"], 1)
        self.assertGreaterEqual(int(row["tasks_completed"]), 1)
        self.assertLess(int(row["steps"]), 64)

    def test_throughput_regime_row_has_zero_budget_flag(self):
        env = _line_env(num_agents=1, horizon=16, n_nodes=4)
        row = run_episode(env, seed=0)
        self.assertEqual(row["task_budget_reached"], 0)
        self.assertEqual(row["steps"], 16)

    def test_summarize_success_regime_block(self):
        rows = [
            {
                "tasks_completed": 200,
                "plan_failures": 0,
                "replans": 10,
                "task_budget_reached": 1,
                "steps": 700,
            },
            {
                "tasks_completed": 150,
                "plan_failures": 2,
                "replans": 12,
                "task_budget_reached": 0,
                "steps": 1024,
            },
        ]
        summary = summarize(rows, task_budget=200)
        self.assertEqual(summary["task_budget"], 200)
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["success_rate"], 0.5)
        lo, hi = summary["success_wilson95"]
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        # Steps-to-budget aggregates only over successful episodes.
        self.assertEqual(summary["steps_to_budget_mean"], 700)
        self.assertEqual(summary["steps_to_budget_max"], 700)


if __name__ == "__main__":
    unittest.main(verbosity=2)
