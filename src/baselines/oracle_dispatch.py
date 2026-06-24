"""Sprint 5 headroom probe — clairvoyant rolling-horizon Hungarian oracle.

Upper bound on **assignment quality** holding Layer 2 fixed: the same LSAP as the
Hungarian baseline, but the candidate pool at step ``t`` also includes tasks
arriving within the next ``K`` steps (known from a pre-recorded arrival
schedule). An agent matched to a *future* task is **held idle** — anticipation:
don't commit it to a worse current task when a closer one is imminent. Only
matches to *currently pending* tasks are committed.

This isolates the **anticipation headroom** a learned dispatcher could capture
beyond myopic-optimal (Hungarian). Used as a go/no-go signal.

CAVEAT (pinned, gate2_metric_decision): the schedule is recorded from a prior
pass; the pool cap can make a different dispatcher's actual spawns diverge
slightly, so this is a *tractable, reproducible* upper bound with K-step
lookahead — not the exact (intractable) offline optimum.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.baselines.hungarian_dispatch import UNREACHABLE_COST


def schedule_from_tasks(all_tasks: Iterable) -> dict[int, list[tuple[str, int]]]:
    """{spawn_step: [(pickup, task_id), ...]} from Task objects of a recorded run."""
    by_step: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for t in all_tasks:
        by_step[t.spawn_step].append((t.pickup, t.id))
    return dict(by_step)


def make_oracle_dispatcher(
    future_by_step: dict[int, list[tuple[str, int]]], lookahead_k: int
) -> Callable:
    """Build a clairvoyant dispatcher (one per seed).

    Uses the TRUE env step from ``ctx.step`` (NOT a self-maintained call counter):
    the env skips the injected dispatcher on steps with no idle agents / no
    pending tasks, so counting invocations would mis-measure the lookahead window
    (it would span more than ``lookahead_k`` env steps). ``future_by_step`` is
    keyed by ``Task.spawn_step``, which is an env step, so the window must use the
    env step too.
    """
    if lookahead_k < 0:
        raise ValueError("lookahead_k must be >= 0")

    def oracle(ctx) -> dict[str, int]:
        t = ctx.step
        idle = ctx.idle_agents
        current = ctx.pending_tasks  # (id, pickup, dropoff)
        if not idle:
            return {}
        current_ids = {tid for (tid, _pk, _do) in current}
        future: list[tuple[int, str]] = []
        for s in range(t + 1, t + lookahead_k + 1):
            future.extend(
                (tid, pk) for (pk, tid) in future_by_step.get(s, [])
                if tid not in current_ids
            )
        cand = [(tid, pk) for (tid, pk, _do) in current] + future  # (task_id, pickup)
        if not cand:
            return {}

        n, m = len(idle), len(cand)
        cost = np.empty((n, m), dtype=float)
        for i in range(n):
            pos = idle[i][1]
            for j in range(m):
                d = ctx.distance(pos, cand[j][1])
                cost[i, j] = d if np.isfinite(d) else UNREACHABLE_COST

        rows, cols = linear_sum_assignment(cost)
        out: dict[str, int] = {}
        for i, j in zip(rows, cols):
            tid = cand[j][0]
            if tid in current_ids and cost[i, j] < UNREACHABLE_COST:
                out[idle[i][0]] = tid
            # else: matched to a future task -> hold the agent (anticipation)
        return out

    return oracle
