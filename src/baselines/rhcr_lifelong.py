"""Sprint 7 (strengthen, R5) — faithful RHCR lifelong reference.

Review R5 flagged the LaCAM "windowed RHCR-style" reference
(``src/baselines/lacam_lifelong``) as not a faithful RHCR: to make LaCAM — a
one-shot joint solver that needs a distinct goal *configuration* — usable in the
lifelong regime, it de-duplicates per-agent waypoints W steps along the shortest
path, pulling an agent whose waypoint is already claimed back off its true path.
That distorts the objective.

This module is the faithful RHCR (Li et al. 2021): rolling-horizon collision
resolution with a **windowed PBS** solver (``src/mapf/windowed_pbs``), the
canonical RHCR low-level. Each replan plans every agent toward its TRUE dispatch
goal, resolves collisions only within a window of ``w`` timesteps, commits the
first ``h < w`` steps, and replans every ``h`` steps. No goal de-duplication —
a shared goal is handled by within-window reservation (a later-priority agent
queues behind whoever holds it), which is the exact lifelong situation the dedup
hack was working around. So the warehouse now has two classical lifelong bars:
windowed-LaCAM (deliberative joint, with the dedup caveat) and this windowed-PBS
RHCR (faithful prioritized).

Reuses ``PP32LifelongController``'s tested execution loop (mask-aware, freeze
sidestep); only planning differs. Standalone runner (own ProcessPool fan-out +
success regime) so Windows-spawn workers construct the controller correctly.

Example (success regime, held-out cell):
    python -m src.baselines.rhcr_lifelong --seeds 0 1 ... 29 --num-agents 20 \
        --task-distribution corner_exchange --task-budget 16 \
        --episode-length 4096 --results-dir results/.../rhcr_n20_success
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

from src.baselines.pp32_lifelong import PP32LifelongController
from src.baselines.stats_utils import wilson_ci
from src.env.warehouse_env import WarehouseEnv
from src.mapf.windowed_pbs import WindowedPBSPlanner
from src.rl.mappo_onpolicy.env_adapter import WarehouseEnvConfig, build_warehouse_env

DEFAULT_WINDOW = 16          # w: collision-resolution horizon
DEFAULT_COMMIT = 8           # h: execution window == replan period (h < w)
DEFAULT_MAX_ORDERS = 4


class RHCRLifelongController(PP32LifelongController):
    """PP-32 execution loop (mask-aware, freeze sidestep) + WINDOWED PBS.

    Faithful RHCR: plan toward true goals, resolve collisions within ``window``
    (w), commit ``commit`` (h) steps, replan every h. ``commit`` is handed to
    the base class as ``replan_horizon`` so the inherited event loop replans on
    the rolling-horizon period (plus the usual dispatch-goal-change triggers).
    """

    def __init__(
        self,
        env: WarehouseEnv,
        window: int = DEFAULT_WINDOW,
        commit: int = DEFAULT_COMMIT,
        max_orders: int = DEFAULT_MAX_ORDERS,
        seed: int = 0,
        reactive_fallback: bool = False,
    ):
        if commit < 1:
            raise ValueError("commit (h) must be >= 1")
        if window < commit:
            raise ValueError("window (w) must be >= commit (h)")
        # h == replan period: the base loop replans when _steps_since_plan >= h.
        super().__init__(env, replan_horizon=commit, seed=seed)
        self.window = window
        self.commit = commit
        # R2/DA parity with the LaCAM reference: a window the solver cannot place
        # (an agent fully blocked under every tried priority order) degrades to a
        # greedy one-step-toward-goal per agent instead of a fleet-wide wait.
        self.reactive_fallback = reactive_fallback
        self.fallback_windows = 0
        self.planner = WindowedPBSPlanner(  # type: ignore[assignment]
            env.G, window=window, max_orders=max_orders, seed=seed
        )

    def _replan(self, goals: dict[str, str]) -> None:
        starts = {a: self.env._agent_info[a].pos for a in self.env.agents}
        result = self.planner.plan(starts, goals)
        self.replans += 1
        self._steps_since_plan = 0
        self._planned_goals = dict(goals)  # trigger on DISPATCH-goal changes too
        self.orders_tried_total += int(result.diagnostics.get("orders_tried", 0))
        self.plan_elapsed_s += float(result.elapsed_s)
        if result.success:
            self._last_plan_failed = False
            # Rolling horizon: keep only the first h steps of the w-step plan;
            # the tail is discarded and recomputed at the next replan.
            self._queues = {
                a: list(path[1 : self.commit + 1]) for a, path in result.paths.items()
            }
        elif self.reactive_fallback:
            self.plan_failures += 1
            self.fallback_windows += 1
            self._last_plan_failed = False
            self._queues = self._greedy_window(starts, goals)
        else:
            self.plan_failures += 1
            self._last_plan_failed = True
            self._queues = {a: [] for a in starts}

    def _greedy_window(
        self, starts: dict[str, str], goals: dict[str, str]
    ) -> dict[str, list[str]]:
        """Next up-to-``commit`` nodes along each agent's shortest path toward
        its goal (current position excluded). Empty if at goal / no path."""
        queues: dict[str, list[str]] = {}
        for a, pos in starts.items():
            g = goals[a]
            if g == pos:
                queues[a] = []
                continue
            p = self.env.router.path(pos, g)
            queues[a] = list(p[1 : self.commit + 1]) if p and len(p) > 1 else []
        return queues


def run_episode(
    env: WarehouseEnv,
    seed: int,
    window: int = DEFAULT_WINDOW,
    commit: int = DEFAULT_COMMIT,
    max_orders: int = DEFAULT_MAX_ORDERS,
    reactive_fallback: bool = False,
) -> dict[str, object]:
    env.reset(seed=seed)
    controller = RHCRLifelongController(
        env,
        window=window,
        commit=commit,
        max_orders=max_orders,
        seed=seed,
        reactive_fallback=reactive_fallback,
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
        "baseline": "rhcr_lifelong",
        "seed": seed,
        "steps": steps,
        "window": window,
        "commit": commit,
        "tasks_completed": tasks_completed,
        "task_budget_reached": int(budget_reached),
        "throughput_per_step": round(tasks_completed / max(steps, 1), 4),
        "replans": controller.replans,
        "plan_failures": controller.plan_failures,
        "fallback_windows": controller.fallback_windows,
        "freeze_sidesteps": controller.freeze_sidesteps,
        "orders_tried_total": controller.orders_tried_total,
        "plan_time_s": round(controller.plan_elapsed_s, 2),
        "validator_interventions": validator_interventions,
        "wall_s": round(elapsed, 1),
    }


_WORKER_ENV: Optional[WarehouseEnv] = None
_WORKER_CFG: Optional[WarehouseEnvConfig] = None


def _run_seed_job(
    cfg: WarehouseEnvConfig,
    seed: int,
    window: int,
    commit: int,
    max_orders: int,
    reactive_fallback: bool = False,
) -> dict[str, object]:
    global _WORKER_ENV, _WORKER_CFG
    if _WORKER_ENV is None or _WORKER_CFG != cfg:
        _WORKER_ENV = build_warehouse_env(cfg)
        _WORKER_CFG = cfg
    return run_episode(
        _WORKER_ENV,
        seed=seed,
        window=window,
        commit=commit,
        max_orders=max_orders,
        reactive_fallback=reactive_fallback,
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
                window=args.window,
                commit=args.commit,
                max_orders=args.max_orders,
                reactive_fallback=args.reactive_fallback,
            )
            rows.append(row)
            print(
                f"[rhcr_lifelong] {i}/{len(args.seeds)} seed={seed} "
                f"tasks={row['tasks_completed']} fails={row['plan_failures']} "
                f"({row['wall_s']}s)",
                flush=True,
            )
        return rows

    print(
        f"[rhcr_lifelong] {len(args.seeds)} seeds across {workers} processes",
        flush=True,
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_seed = {
            pool.submit(
                _run_seed_job,
                cfg,
                seed,
                args.window,
                args.commit,
                args.max_orders,
                args.reactive_fallback,
            ): seed
            for seed in args.seeds
        }
        done = 0
        for future in concurrent.futures.as_completed(future_to_seed):
            row = future.result()
            rows.append(row)
            done += 1
            print(
                f"[rhcr_lifelong] {done}/{len(args.seeds)} seed={row['seed']} "
                f"tasks={row['tasks_completed']} fails={row['plan_failures']} "
                f"({row['wall_s']}s)",
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
        "baseline": "rhcr_lifelong",
        "seeds": len(tasks),
        "tasks_completed_mean": round(mean, 2),
        "tasks_completed_std": round(std, 2),
        "plan_failures_total": sum(int(r["plan_failures"]) for r in rows),
        "fallback_windows_total": sum(int(r.get("fallback_windows", 0)) for r in rows),
        "replans_total": sum(int(r["replans"]) for r in rows),
    }
    if task_budget is not None:
        n = len(rows)
        succ = sum(int(r["task_budget_reached"]) for r in rows)
        lo, hi = wilson_ci(succ, n)
        steps_succ = [int(r["steps"]) for r in rows if int(r["task_budget_reached"])]
        summary.update(
            {
                "task_budget": task_budget,
                "success_count": succ,
                "success_rate": round(succ / n, 3),
                "success_wilson95": [round(lo, 3), round(hi, 3)],
                "steps_to_budget_mean": (
                    round(statistics.mean(steps_succ), 1) if steps_succ else None
                ),
            }
        )
    return summary


def write_outputs(rows, summary, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = results_dir / "rhcr_lifelong_runs.csv"
    with runs_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (results_dir / "rhcr_lifelong_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[rhcr_lifelong] wrote {runs_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(30)))
    parser.add_argument("--num-agents", type=int, default=20)
    parser.add_argument("--episode-length", type=int, default=1024)
    parser.add_argument("--task-rate", type=float, default=0.1)
    parser.add_argument("--task-distribution", default="uniform")
    parser.add_argument("--task-budget", type=int, default=None)
    parser.add_argument(
        "--window", type=int, default=DEFAULT_WINDOW,
        help="w: collision-resolution horizon for the windowed PBS solver.",
    )
    parser.add_argument(
        "--commit", type=int, default=DEFAULT_COMMIT,
        help="h: execution window == replan period (must be <= w). RHCR commits "
        "the first h steps of each w-step plan, then replans.",
    )
    parser.add_argument("--max-orders", type=int, default=DEFAULT_MAX_ORDERS)
    parser.add_argument("--no-lookahead-mask", action="store_true")
    parser.add_argument(
        "--reactive-fallback", action="store_true",
        help="RHCR-style: on a window the PBS solver cannot place, fall back to "
        "greedy one-step-toward-goal per agent instead of freezing the fleet "
        "(parity with the LaCAM reference's --reactive-fallback).",
    )
    parser.add_argument(
        "--parallel", type=int, default=max(1, (os.cpu_count() or 4) - 2)
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results") / "sprint7_strengthen" / "rhcr_lifelong",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    rows = run_reference(args)
    summary = summarize(rows, task_budget=args.task_budget)
    write_outputs(rows, summary, args.results_dir)
    print(
        f"\n[rhcr_lifelong] tasks_completed = {summary['tasks_completed_mean']} "
        f"+/- {summary['tasks_completed_std']} over {summary['seeds']} seeds"
    )
    if args.task_budget is not None:
        lo, hi = summary["success_wilson95"]  # type: ignore[misc]
        print(
            f"[rhcr_lifelong] success regime: budget={args.task_budget} "
            f"-> {summary['success_count']}/{summary['seeds']} succeeded "
            f"(rate {summary['success_rate']}, Wilson95 [{lo}, {hi}])"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
