"""Tests for the Sprint 5 clairvoyant oracle dispatcher."""

from __future__ import annotations

import unittest

from src.baselines.hungarian_dispatch import hungarian_dispatch
from src.baselines.oracle_dispatch import make_oracle_dispatcher, schedule_from_tasks
from src.env.warehouse_env import DispatchContext


def _ctx(idle, tasks, dist, step=0):
    return DispatchContext(
        idle_agents=tuple(idle), pending_tasks=tuple(tasks), distance=dist, step=step
    )


class TestOracleDispatch(unittest.TestCase):
    def test_holds_agent_for_imminent_closer_task(self):
        # Current task is far (d=10); a closer task (d=1) arrives next step.
        # Anticipation: oracle should HOLD the agent (assign nothing now),
        # whereas Hungarian would commit it to the far current task.
        D = {("a", "far"): 10.0, ("a", "near"): 1.0}
        ctx = _ctx([("A", "a", 0.0)], [(1, "far", "x")], lambda s, t: D[(s, t)])
        # task id 2 ("near") spawns at step 1
        oracle = make_oracle_dispatcher({1: [("near", 2)]}, lookahead_k=2)
        self.assertEqual(oracle(ctx), {})  # held (first call -> t=0, sees step 1)
        # Hungarian (myopic) would grab the far current task
        self.assertEqual(hungarian_dispatch(ctx), {"A": 1})

    def test_commits_current_when_no_better_future(self):
        # Current task near (d=1); future task far (d=100) -> take current now.
        D = {("a", "near"): 1.0, ("a", "far"): 100.0}
        ctx = _ctx([("A", "a", 0.0)], [(1, "near", "x")], lambda s, t: D[(s, t)])
        oracle = make_oracle_dispatcher({1: [("far", 2)]}, lookahead_k=2)
        self.assertEqual(oracle(ctx), {"A": 1})

    def test_lookahead_zero_equals_myopic(self):
        # With K=0 the oracle sees no future -> same as Hungarian on current.
        D = {("a", "far"): 10.0}
        ctx = _ctx([("A", "a", 0.0)], [(1, "far", "x")], lambda s, t: D[(s, t)])
        oracle = make_oracle_dispatcher({1: [("near", 2)]}, lookahead_k=0)
        self.assertEqual(oracle(ctx), {"A": 1})

    def test_window_uses_ctx_step_not_call_count(self):
        # REGRESSION: the lookahead window is driven by ctx.step (the true env
        # step), NOT by how many times the oracle was called (the env skips the
        # dispatcher on empty steps, so a call-counter would mis-measure K).
        D = {("a", "far"): 10.0, ("a", "near"): 1.0}
        dist = lambda s, t: D[(s, t)]  # noqa: E731
        idle, cur = [("A", "a", 0.0)], [(1, "far", "x")]
        oracle = make_oracle_dispatcher({1: [("near", 2)]}, lookahead_k=1)
        # step=0: window (0,1] includes "near" (spawns at step 1) -> hold
        self.assertEqual(oracle(_ctx(idle, cur, dist, step=0)), {})
        # step=1: window (1,2] no longer includes it -> commit current
        self.assertEqual(oracle(_ctx(idle, cur, dist, step=1)), {"A": 1})
        # repeated calls at the SAME step do NOT slide the window (no counter)
        self.assertEqual(oracle(_ctx(idle, cur, dist, step=0)), {})
        self.assertEqual(oracle(_ctx(idle, cur, dist, step=0)), {})

    def test_schedule_from_tasks(self):
        class _T:
            def __init__(self, spawn_step, pickup, tid):
                self.spawn_step, self.pickup, self.id = spawn_step, pickup, tid

        sched = schedule_from_tasks([_T(0, "p", 1), _T(0, "q", 2), _T(3, "r", 3)])
        self.assertEqual(set(sched[0]), {("p", 1), ("q", 2)})
        self.assertEqual(sched[3], [("r", 3)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
