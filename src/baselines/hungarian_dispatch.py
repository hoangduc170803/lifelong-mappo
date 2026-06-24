"""Sprint 5 Bậc 0 — Hungarian (LSAP) dispatcher for the WarehouseEnv hook.

Rerun-on-event optimal task→agent assignment. At each dispatch event the env
hands a ``DispatchContext`` (idle agents, pending tasks, distance fn); this
solves the optimal one-to-one assignment minimizing total pickup distance via
``scipy.optimize.linear_sum_assignment`` and returns ``{agent_name: task_id}``.

Role in Sprint 5:
- the **GATE 2 baseline + fallback** dispatcher (vs the learned single-pick
  policy), run on top of the same frozen Layer 2 router;
- the base for the **clairvoyant headroom oracle**, which is this same LSAP with
  a K-step arrival lookahead added to the cost (see headroom probe).

Plug in via ``env.set_dispatcher(hungarian_dispatch)`` (or the constructor).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from src.env.warehouse_env import DispatchContext

# Finite stand-in for an unreachable (inf-distance) agent→pickup pair: large
# enough that LSAP never prefers it, but finite so the solver stays well-posed.
UNREACHABLE_COST = 1e9


def hungarian_dispatch(ctx: "DispatchContext") -> dict[str, int]:
    """Optimal one-to-one assignment minimizing total agent→pickup distance.

    Rectangular instances (more idle agents than pending tasks, or vice versa)
    are handled by ``linear_sum_assignment`` — it matches ``min(n, m)`` pairs.
    Pairs whose pickup is unreachable are never assigned.
    """
    idle = ctx.idle_agents          # (name, pos, age_idle)
    tasks = ctx.pending_tasks       # (task_id, pickup, dropoff)
    if not idle or not tasks:
        return {}

    n, m = len(idle), len(tasks)
    cost = np.empty((n, m), dtype=float)
    for i in range(n):
        pos = idle[i][1]
        for j in range(m):
            d = ctx.distance(pos, tasks[j][1])
            cost[i, j] = d if np.isfinite(d) else UNREACHABLE_COST

    rows, cols = linear_sum_assignment(cost)
    assignment: dict[str, int] = {}
    for i, j in zip(rows, cols):
        if cost[i, j] >= UNREACHABLE_COST:
            continue  # do not dispatch an agent to an unreachable pickup
        assignment[idle[i][0]] = tasks[j][0]
    return assignment
