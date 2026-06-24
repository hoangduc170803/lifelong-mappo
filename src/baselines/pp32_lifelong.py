"""Sprint 4a D5 — PP-32 lifelong reference on the warehouse env.

Drives the SAME lifelong ``WarehouseEnv`` used for MAPPO training (greedy
dispatch, Poisson arrivals, pool cap, safety validator, compass actions) with
the PP-32 classical planner replacing the learned policy. This produces the
apples-to-apples denominator for the D4 reward-sweep criterion
``tasks_completed >= 0.9 x PP-32`` on 5-AGV uniform lifelong:

- same env construction as MAPPO training (`build_warehouse_env`)
- same metric: ``tasks_completed_total`` at the end of a fixed-horizon episode
- same map / agent count / task rate / episode length

Control loop: rolling-horizon PP-32 (Sprint 3.5 gate memo §3.2.4 convention,
H=20) plus event-triggered replans whenever any agent's dispatch goal changes
(pickup reached -> goal flips to dropoff; dropoff reached -> greedy dispatch
assigns a new task) or a path queue runs dry before its goal. Validator
forced-waits leave the queue un-advanced so the agent retries the same node
next tick; periodic replanning bounds how stale a plan can get.

Example (full D5 reference):
    python -m src.baselines.pp32_lifelong \
        --seeds 0 1 2 3 4 5 6 7 8 9 --episode-length 1024 \
        --results-dir results/sprint4a/pp32_lifelong

Smoke:
    python -m src.baselines.pp32_lifelong --seeds 0 --episode-length 128 \
        --results-dir results/sprint4a/pp32_lifelong_smoke
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Optional, Sequence

from src.baselines.stats_utils import wilson_ci
from src.env.compass_mapper import WAIT_SLOT
from src.env.warehouse_env import WarehouseEnv
from src.mapf.priority_search import PrioritySearchPlanner
from src.rl.mappo_onpolicy.env_adapter import (
    WarehouseEnvConfig,
    build_warehouse_env,
)

DEFAULT_REPLAN_HORIZON = 20
# Rolling-horizon legs are agent->goal paths, far shorter than the one-shot
# 512-step budget used by the Sprint 3.5 gate. A tighter budget halves the
# cost of a *failed* PP-32 attempt (failures explore the full space-time
# graph up to max_time before giving up).
DEFAULT_MAX_TIME = 256


class PP32LifelongController:
    """Action provider: PP-32 rolling-horizon plans -> compass action slots.

    The env keeps full authority over dispatch, task lifecycle, and safety
    validation — exactly as it does for the MAPPO policy. The controller only
    answers "which compass slot should each agent take this tick".
    """

    def __init__(
        self,
        env: WarehouseEnv,
        max_orders: int = 32,
        replan_horizon: int = DEFAULT_REPLAN_HORIZON,
        max_time: int = DEFAULT_MAX_TIME,
        seed: int = 0,
    ):
        if replan_horizon < 1:
            raise ValueError("replan_horizon must be >= 1")
        self.env = env
        self.replan_horizon = replan_horizon
        self.seed = seed
        self.planner = PrioritySearchPlanner(
            env.G, max_time=max_time, max_orders=max_orders
        )
        # Remaining planned nodes per agent (current position excluded).
        self._queues: dict[str, list[str]] = {}
        self._planned_goals: dict[str, str] = {}
        self._steps_since_plan = replan_horizon  # force plan on first tick
        # Failure backoff: when a replan fails, empty queues must NOT trigger
        # an immediate retry next tick (a failed PP-32 attempt costs 32 orders
        # of space-time A*; retrying every tick livelocks the episode — seen
        # on seed 0 of the first reference run: 155 failures, 6000 s). Retry
        # only when goals change or the periodic horizon elapses.
        self._last_plan_failed = False
        # Deadlock recovery: the env's lookahead action mask reserves every
        # other agent's position AND predicted-next node with no tie-breaking,
        # so two agents whose shortest paths contend for the same free node
        # block each other symmetrically and forever (seen on seeds 2/5/6/7 of
        # the first reference run: convoys frozen behind a mutual claim,
        # 6-10 tasks over 1024 steps). Priority-order perturbation does NOT
        # help — shortest-path geometry is order-independent. Instead the
        # controller is mask-aware (it reads the same action mask MAPPO gets
        # in its observation): a blocked agent waits, and any agent whose
        # planned move stays mask-blocked for `freeze_patience` consecutive
        # ticks sidesteps to an unmasked neighbor (one agent per tick, lowest
        # name, deterministic) to break the symmetry; its queue is dropped so
        # the event trigger re-routes it. Per-agent streaks matter: a global
        # all-static detector misses convoys frozen behind one mover (seen on
        # seeds 2/5 of the second run: detector never fired).
        self.freeze_patience = 5
        self._blocked_ticks: dict[str, int] = {}

        self.replans = 0
        self.plan_failures = 0
        self.freeze_sidesteps = 0
        self.orders_tried_total = 0
        self.plan_elapsed_s = 0.0

    # ------------------------------------------------------------------ state

    def _current_goals(self) -> dict[str, str]:
        """Each agent's dispatch goal right now; idle agents hold position."""
        goals: dict[str, str] = {}
        for a in self.env.agents:
            info = self.env._agent_info[a]
            goals[a] = info.task.current_goal if info.task is not None else info.pos
        return goals

    def _need_replan(self, goals: dict[str, str]) -> bool:
        if self._steps_since_plan >= self.replan_horizon:
            return True
        if goals != self._planned_goals:
            return True
        if self._last_plan_failed:
            # Backoff: queues are intentionally empty after a failure; wait
            # for a goal change or the periodic horizon instead of burning a
            # full 32-order PP attempt every tick.
            return False
        for a, goal in goals.items():
            info = self.env._agent_info[a]
            if info.pos != goal and not self._queues.get(a):
                return True
        return False

    def _replan(self, goals: dict[str, str]) -> None:
        starts = {a: self.env._agent_info[a].pos for a in self.env.agents}
        result = self.planner.plan(starts, goals, seed=self.seed)
        self.replans += 1
        self._steps_since_plan = 0
        self._planned_goals = dict(goals)
        self.orders_tried_total += int(result.diagnostics.get("orders_tried", 0))
        self.plan_elapsed_s += float(result.elapsed_s)
        if result.success:
            self._last_plan_failed = False
            self._queues = {
                a: list(path[1:]) for a, path in result.paths.items()
            }
        else:
            # All agents hold position until the next trigger (goal change or
            # periodic H). Counted so the reference reports planner fragility.
            self.plan_failures += 1
            self._last_plan_failed = True
            self._queues = {a: [] for a in starts}

    # ------------------------------------------------------------------ API

    def actions(self) -> dict[str, int]:
        """Compass action for every live agent this tick."""
        goals = self._current_goals()
        if self._need_replan(goals):
            self._replan(goals)
        # Same per-agent action mask MAPPO receives in its observation; a
        # masked slot would be sanitized to WAIT by the env anyway, but
        # knowing WHO is blocked lets the freeze breaker pick a sidestep.
        masks, slot_maps, _ = self.env._action_masks_for_agents(self.env.agents)
        acts: dict[str, int] = {}
        blocked: list[str] = []
        for a in self.env.agents:
            info = self.env._agent_info[a]
            queue = self._queues.get(a, [])
            # Consume planned in-place waits (space-time path repeats a node).
            while queue and queue[0] == info.pos:
                queue.pop(0)
            if not queue:
                acts[a] = WAIT_SLOT
                continue
            slot = self._slot_to(info.pos, queue[0])
            if slot is None:
                # Planned next node is not a compass neighbor — should not
                # happen on a consistent graph; drop the queue so the event
                # trigger replans next tick.
                self._queues[a] = []
                acts[a] = WAIT_SLOT
            elif not masks[a][slot]:
                # Lookahead mask blocks the planned move (occupied node or a
                # contended prediction). Wait and retry; queue is kept.
                acts[a] = WAIT_SLOT
                blocked.append(a)
            else:
                acts[a] = slot

        # Update per-agent blocked streaks; agents that move or have no plan
        # reset to zero.
        for a in self.env.agents:
            if a in blocked:
                self._blocked_ticks[a] = self._blocked_ticks.get(a, 0) + 1
            else:
                self._blocked_ticks[a] = 0

        stuck = sorted(
            a for a in blocked
            if self._blocked_ticks[a] >= self.freeze_patience
        )
        if stuck:
            # Symmetric mutual block: let exactly one stuck agent (lowest
            # name, deterministic) sidestep to any unmasked neighbor. Its
            # queue is dropped so the event trigger re-routes from the new
            # position next tick.
            agent = stuck[0]
            valid = [
                slot
                for slot in slot_maps[agent]
                if slot != WAIT_SLOT and masks[agent][slot]
            ]
            if valid:
                acts[agent] = int(valid[0])
                self._queues[agent] = []
                self.freeze_sidesteps += 1
                self._blocked_ticks[agent] = 0

        self._steps_since_plan += 1
        return acts

    def observe_transition(self) -> None:
        """Advance queues for agents that actually reached their planned node.

        Agents the validator forced to wait keep their queue head and retry
        the same node next tick.
        """
        for a in self.env.agents:
            info = self.env._agent_info[a]
            queue = self._queues.get(a)
            if queue and info.pos == queue[0]:
                queue.pop(0)

    def _slot_to(self, src: str, target: str) -> Optional[int]:
        _, slot_to_node = self.env.compass.get(src)
        for slot, node in slot_to_node.items():
            if node == target:
                return int(slot)
        return None


# ---------------------------------------------------------------------- runner


def run_episode(
    env: WarehouseEnv,
    seed: int,
    max_orders: int = 32,
    replan_horizon: int = DEFAULT_REPLAN_HORIZON,
    max_time: int = DEFAULT_MAX_TIME,
) -> dict[str, object]:
    """One fixed-horizon lifelong episode under PP-32 control."""
    env.reset(seed=seed)
    controller = PP32LifelongController(
        env,
        max_orders=max_orders,
        replan_horizon=replan_horizon,
        max_time=max_time,
        seed=seed,
    )
    started = time.perf_counter()
    tasks_completed = 0
    validator_interventions = 0
    budget_reached = False
    steps = 0
    while env.agents:
        acts = controller.actions()
        _, _, terminated, truncated, infos = env.step(acts)
        controller.observe_transition()
        steps += 1
        sample = infos[next(iter(infos))]
        tasks_completed = int(sample["tasks_completed_total"])
        validator_interventions += int(sample["validator_interventions"])
        budget_reached = bool(sample["task_budget_reached"])
        if all(terminated.values()) or all(truncated.values()):
            break
    elapsed = time.perf_counter() - started
    return {
        "baseline": "pp32_lifelong",
        "seed": seed,
        "steps": steps,
        "tasks_completed": tasks_completed,
        "task_budget_reached": int(budget_reached),
        "throughput_per_step": round(tasks_completed / max(steps, 1), 4),
        "replans": controller.replans,
        "plan_failures": controller.plan_failures,
        "freeze_sidesteps": controller.freeze_sidesteps,
        "orders_tried_total": controller.orders_tried_total,
        "plan_time_s": round(controller.plan_elapsed_s, 2),
        "validator_interventions": validator_interventions,
        "wall_s": round(elapsed, 1),
    }


# Per-process env for the parallel path. Built lazily by each worker so the
# (graph parse + all-pairs router) cost is paid once per process, not per
# seed. Module-level because Windows spawn pickles only the function name.
_WORKER_ENV: Optional[WarehouseEnv] = None
_WORKER_CFG: Optional[WarehouseEnvConfig] = None


def _run_seed_job(
    cfg: WarehouseEnvConfig,
    seed: int,
    max_orders: int,
    replan_horizon: int,
    max_time: int,
) -> dict[str, object]:
    global _WORKER_ENV, _WORKER_CFG
    if _WORKER_ENV is None or _WORKER_CFG != cfg:
        _WORKER_ENV = build_warehouse_env(cfg)
        _WORKER_CFG = cfg
    return run_episode(
        _WORKER_ENV,
        seed=seed,
        max_orders=max_orders,
        replan_horizon=replan_horizon,
        max_time=max_time,
    )


def run_reference(args: argparse.Namespace) -> list[dict[str, object]]:
    cfg = WarehouseEnvConfig(
        num_agents=args.num_agents,
        episode_horizon=args.episode_length,
        task_rate=args.task_rate,
        task_distribution=args.task_distribution,
        lookahead_action_mask=not args.no_lookahead_mask,
        task_budget=args.task_budget,
        seed=args.seeds[0],
    )
    rows: list[dict[str, object]] = []
    workers = max(1, min(args.parallel, len(args.seeds)))
    if workers == 1:
        env = build_warehouse_env(cfg)
        for i, seed in enumerate(args.seeds, start=1):
            row = run_episode(
                env,
                seed=seed,
                max_orders=args.max_orders,
                replan_horizon=args.replan_horizon,
                max_time=args.max_time,
            )
            rows.append(row)
            print(
                f"[pp32_lifelong] {i}/{len(args.seeds)} seed={seed} "
                f"tasks={row['tasks_completed']} replans={row['replans']} "
                f"failures={row['plan_failures']} ({row['wall_s']}s)",
                flush=True,
            )
        return rows

    print(
        f"[pp32_lifelong] {len(args.seeds)} seeds across {workers} processes",
        flush=True,
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_seed = {
            pool.submit(
                _run_seed_job,
                cfg,
                seed,
                args.max_orders,
                args.replan_horizon,
                args.max_time,
            ): seed
            for seed in args.seeds
        }
        done = 0
        for future in concurrent.futures.as_completed(future_to_seed):
            row = future.result()
            rows.append(row)
            done += 1
            print(
                f"[pp32_lifelong] {done}/{len(args.seeds)} seed={row['seed']} "
                f"tasks={row['tasks_completed']} replans={row['replans']} "
                f"failures={row['plan_failures']} ({row['wall_s']}s)",
                flush=True,
            )
    rows.sort(key=lambda r: int(r["seed"]))  # type: ignore[arg-type]
    return rows


def summarize(
    rows: Sequence[dict[str, object]],
    task_budget: Optional[int] = None,
) -> dict[str, object]:
    tasks = [float(r["tasks_completed"]) for r in rows]
    mean = statistics.mean(tasks) if tasks else float("nan")
    std = statistics.pstdev(tasks) if len(tasks) > 1 else 0.0
    summary: dict[str, object] = {
        "baseline": "pp32_lifelong",
        "seeds": len(tasks),
        "tasks_completed_mean": round(mean, 2),
        "tasks_completed_std": round(std, 2),
        "tasks_completed_min": min(tasks) if tasks else None,
        "tasks_completed_max": max(tasks) if tasks else None,
        "d4_criterion_0.9x": round(0.9 * mean, 2),
        "plan_failures_total": sum(int(r["plan_failures"]) for r in rows),
        "replans_total": sum(int(r["replans"]) for r in rows),
    }
    if task_budget is not None:
        summary.update(_success_regime_summary(rows, task_budget))
    return summary


def _success_regime_summary(
    rows: Sequence[dict[str, object]], task_budget: int
) -> dict[str, object]:
    """Success-regime block (GATE 1): episode succeeds iff the env reached
    its `task_budget` terminal before the step cap. Wilson CI per PLAN's
    statistical reporting rules; steps-to-budget only over successes."""
    n = len(rows)
    successes = sum(int(r["task_budget_reached"]) for r in rows)
    lo, hi = wilson_ci(successes, n)
    steps_succ = [int(r["steps"]) for r in rows if int(r["task_budget_reached"])]
    return {
        "task_budget": task_budget,
        "success_count": successes,
        "success_rate": round(successes / n, 3),
        "success_wilson95": [round(lo, 3), round(hi, 3)],
        "steps_to_budget_mean": (
            round(statistics.mean(steps_succ), 1) if steps_succ else None
        ),
        "steps_to_budget_max": max(steps_succ) if steps_succ else None,
    }


def write_outputs(
    rows: Sequence[dict[str, object]],
    summary: dict[str, object],
    results_dir: Path,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = results_dir / "pp32_lifelong_runs.csv"
    with runs_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_json = results_dir / "pp32_lifelong_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[pp32_lifelong] wrote {runs_csv}")
    print(f"[pp32_lifelong] wrote {summary_json}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--episode-length", type=int, default=1024)
    parser.add_argument("--task-rate", type=float, default=0.1)
    parser.add_argument("--task-distribution", default="uniform")
    parser.add_argument(
        "--task-budget",
        type=int,
        default=None,
        help=(
            "Success regime (GATE 1): episode terminates once this many tasks "
            "complete; --episode-length becomes the step cap. Adds success "
            "rate + Wilson 95%% CI to the summary. Default: throughput regime "
            "(fixed horizon, no terminal)."
        ),
    )
    parser.add_argument("--max-orders", type=int, default=32)
    parser.add_argument("--replan-horizon", type=int, default=DEFAULT_REPLAN_HORIZON)
    parser.add_argument("--max-time", type=int, default=DEFAULT_MAX_TIME)
    parser.add_argument(
        "--parallel",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 2),
        help="Worker processes for seed fan-out (default: cpu_count - 2).",
    )
    parser.add_argument(
        "--no-lookahead-mask",
        action="store_true",
        help=(
            "Disable the env's lookahead action mask (PP-native semantics; "
            "safety validator still prevents collisions). Use as a secondary "
            "upper-bracket reference — MAPPO trains WITH the mask."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results") / "sprint4a" / "pp32_lifelong",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    rows = run_reference(args)
    summary = summarize(rows, task_budget=args.task_budget)
    write_outputs(rows, summary, args.results_dir)
    print(
        f"\n[pp32_lifelong] reference: tasks_completed = "
        f"{summary['tasks_completed_mean']} +/- {summary['tasks_completed_std']} "
        f"over {summary['seeds']} seeds "
        f"-> D4 0.9x criterion = {summary['d4_criterion_0.9x']}"
    )
    if args.task_budget is not None:
        lo, hi = summary["success_wilson95"]  # type: ignore[misc]
        print(
            f"[pp32_lifelong] success regime: budget={args.task_budget} "
            f"-> {summary['success_count']}/{summary['seeds']} succeeded "
            f"(rate {summary['success_rate']}, Wilson95 [{lo}, {hi}])"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
