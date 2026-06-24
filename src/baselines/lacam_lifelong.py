"""Sprint 4b P0 — LaCAM lifelong reference (strong deliberative classical bar).

Review (2026-06-14) raised a CRITICAL attribution issue: the GATE-1 "beats
classical" claim rested on PP-32 (collapses) and *one-step greedy* PIBT
(freezes) — both reactive. The honest question: does a STRONG deliberative
joint planner — LaCAM (Okumura 2023), which solves these instances one-shot at
100% (P3) — also gridlock under sustained N=20 load, or is the freeze specific
to one-step controllers?

This drives the SAME lifelong env as the other references using **windowed
(RHCR-style) LaCAM**: each replan targets per-agent waypoints W steps along the
shortest path to the dispatch goal, de-duplicated to a feasible config, so
LaCAM solves a short-horizon distinct-goal MAPF (the regime where it succeeds).
NOTE: an earlier version planned to the FULL goal config and failed spuriously
when lifelong dispatch gave two agents the same goal node (impossible to occupy
simultaneously) — see `results/sprint4b/p0_strong_baseline.md` RETRACTED. The
windowed formulation fixes that. Reuses `PP32LifelongController`'s tested
execution loop (mask-aware, freeze sidestep).

Standalone runner (own ProcessPool fan-out + success regime) so Windows-spawn
workers construct the LaCAM controller correctly.

Example (success regime):
    python -m src.baselines.lacam_lifelong --seeds 0 1 ... 29 --num-agents 20 \
        --task-distribution hotspot --task-budget 11 --episode-length 4096 \
        --results-dir results/sprint4b/lacam_lifelong_hotspot_n20_success
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
from src.mapf.lacam import LaCAMPlanner
from src.rl.mappo_onpolicy.env_adapter import WarehouseEnvConfig, build_warehouse_env

DEFAULT_MAX_EXPANSIONS = 30_000
DEFAULT_MAX_TIME_S = 10.0
DEFAULT_REPLAN_HORIZON = 20


class LaCAMLifelongController(PP32LifelongController):
    """PP-32 execution loop (mask-aware, freeze sidestep) + WINDOWED LaCAM.

    RHCR-style (Li et al. 2021) bounded-horizon planning: instead of solving
    the full goal configuration (impossible when lifelong dispatch gives two
    agents the same goal node — that bug sank the first attempt), each replan
    targets a per-agent **waypoint** W steps along the agent's shortest path to
    its current dispatch goal. Waypoints are de-duplicated (an agent whose
    waypoint is already claimed pulls back along its own path) so the target
    config LaCAM receives is always feasible (distinct nodes). LaCAM then solves
    a short-horizon distinct-goal MAPF — the easy regime where it succeeds (P3).
    Execute the window, replan; agents flow toward shared goals across windows
    without ever being required to occupy a node simultaneously.
    """

    def __init__(
        self,
        env: WarehouseEnv,
        replan_horizon: int = DEFAULT_REPLAN_HORIZON,
        max_expansions: int = DEFAULT_MAX_EXPANSIONS,
        max_time_s: float = DEFAULT_MAX_TIME_S,
        seed: int = 0,
        reactive_fallback: bool = False,
    ):
        super().__init__(env, replan_horizon=replan_horizon, seed=seed)
        self.window = replan_horizon
        # Review R2/DA: the bare windowed planner freezes the WHOLE fleet on any
        # window it cannot solve in budget (all-or-nothing). A production RHCR
        # has a reactive fallback so agents keep moving. With this flag ON, a
        # failed window degrades to a greedy one-step-toward-goal per agent
        # (mask-aware via the inherited execution loop + freeze sidestep) instead
        # of a fleet-wide wait. Default OFF preserves the P0 numbers.
        self.reactive_fallback = reactive_fallback
        self.fallback_windows = 0
        self.planner = LaCAMPlanner(  # type: ignore[assignment]
            env.G,
            env.router.distance,
            seed=seed,
            max_expansions=max_expansions,
            max_time_s=max_time_s,
        )

    def _windowed_waypoints(
        self, goals: dict[str, str]
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Per-agent target W steps toward its goal, de-duplicated to a feasible
        (distinct-node) configuration for LaCAM."""
        agents = list(self.env.agents)
        positions = {a: self.env._agent_info[a].pos for a in agents}
        paths: dict[str, list[str]] = {}
        for a in agents:
            g = goals[a]
            if g == positions[a]:
                paths[a] = [positions[a]]
            else:
                p = self.env.router.path(positions[a], g)
                paths[a] = p if p else [positions[a]]
        # Closest-to-goal agents claim their waypoint first; a contested
        # waypoint pulls the further agent back along its own path (positions
        # are distinct, so index 0 is always free -> termination guaranteed).
        order = sorted(agents, key=lambda a: len(paths[a]))
        claimed: dict[str, str] = {}
        waypoints: dict[str, str] = {}
        for a in order:
            p = paths[a]
            idx = min(self.window, len(p) - 1)
            while idx > 0 and p[idx] in claimed:
                idx -= 1
            wp = p[idx]
            waypoints[a] = wp
            claimed[wp] = a
        return positions, waypoints

    def _replan(self, goals: dict[str, str]) -> None:
        starts, waypoints = self._windowed_waypoints(goals)
        result = self.planner.plan(starts, waypoints)  # short-horizon, distinct
        self.replans += 1
        self._steps_since_plan = 0
        self._planned_goals = dict(goals)  # trigger on DISPATCH-goal changes
        self.orders_tried_total += int(result.diagnostics.get("expansions", 0))
        self.plan_elapsed_s += float(result.elapsed_s)
        if result.success:
            self._last_plan_failed = False
            self._queues = {a: list(path[1:]) for a, path in result.paths.items()}
        elif self.reactive_fallback:
            # RHCR-style fallback: don't freeze the fleet. Hand each agent a
            # greedy window toward its goal; the inherited mask-aware loop walks
            # it one step at a time (waiting/sidestepping on contention). Not a
            # dead wait, so clear the backoff flag — when these greedy queues
            # drain (~window ticks) the normal trigger re-attempts LaCAM.
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
        """Next up-to-``window`` nodes along each agent's shortest path toward
        its goal (current position excluded). Empty if at goal / no path."""
        queues: dict[str, list[str]] = {}
        for a, pos in starts.items():
            g = goals[a]
            if g == pos:
                queues[a] = []
                continue
            p = self.env.router.path(pos, g)
            queues[a] = list(p[1 : self.window + 1]) if p and len(p) > 1 else []
        return queues


def run_episode(
    env: WarehouseEnv,
    seed: int,
    replan_horizon: int = DEFAULT_REPLAN_HORIZON,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS,
    max_time_s: float = DEFAULT_MAX_TIME_S,
    reactive_fallback: bool = False,
) -> dict[str, object]:
    env.reset(seed=seed)
    controller = LaCAMLifelongController(
        env,
        replan_horizon=replan_horizon,
        max_expansions=max_expansions,
        max_time_s=max_time_s,
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
        "baseline": "lacam_lifelong",
        "seed": seed,
        "steps": steps,
        "tasks_completed": tasks_completed,
        "task_budget_reached": int(budget_reached),
        "throughput_per_step": round(tasks_completed / max(steps, 1), 4),
        "replans": controller.replans,
        "plan_failures": controller.plan_failures,
        "fallback_windows": controller.fallback_windows,
        "freeze_sidesteps": controller.freeze_sidesteps,
        "expansions_total": controller.orders_tried_total,
        "plan_time_s": round(controller.plan_elapsed_s, 2),
        "validator_interventions": validator_interventions,
        "wall_s": round(elapsed, 1),
    }


_WORKER_ENV: Optional[WarehouseEnv] = None
_WORKER_CFG: Optional[WarehouseEnvConfig] = None


def _run_seed_job(
    cfg: WarehouseEnvConfig,
    seed: int,
    replan_horizon: int,
    max_expansions: int,
    max_time_s: float,
    reactive_fallback: bool = False,
) -> dict[str, object]:
    global _WORKER_ENV, _WORKER_CFG
    if _WORKER_ENV is None or _WORKER_CFG != cfg:
        _WORKER_ENV = build_warehouse_env(cfg)
        _WORKER_CFG = cfg
    return run_episode(
        _WORKER_ENV,
        seed=seed,
        replan_horizon=replan_horizon,
        max_expansions=max_expansions,
        max_time_s=max_time_s,
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
                replan_horizon=args.replan_horizon,
                max_expansions=args.max_expansions,
                max_time_s=args.max_time_s,
                reactive_fallback=args.reactive_fallback,
            )
            rows.append(row)
            print(
                f"[lacam_lifelong] {i}/{len(args.seeds)} seed={seed} "
                f"tasks={row['tasks_completed']} fails={row['plan_failures']} "
                f"({row['wall_s']}s)",
                flush=True,
            )
        return rows

    print(
        f"[lacam_lifelong] {len(args.seeds)} seeds across {workers} processes",
        flush=True,
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_seed = {
            pool.submit(
                _run_seed_job,
                cfg,
                seed,
                args.replan_horizon,
                args.max_expansions,
                args.max_time_s,
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
                f"[lacam_lifelong] {done}/{len(args.seeds)} seed={row['seed']} "
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
        "baseline": "lacam_lifelong",
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
    runs_csv = results_dir / "lacam_lifelong_runs.csv"
    with runs_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (results_dir / "lacam_lifelong_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[lacam_lifelong] wrote {runs_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(30)))
    parser.add_argument("--num-agents", type=int, default=20)
    parser.add_argument("--episode-length", type=int, default=1024)
    parser.add_argument("--task-rate", type=float, default=0.1)
    parser.add_argument("--task-distribution", default="uniform")
    parser.add_argument("--task-budget", type=int, default=None)
    parser.add_argument("--replan-horizon", type=int, default=DEFAULT_REPLAN_HORIZON)
    parser.add_argument("--max-expansions", type=int, default=DEFAULT_MAX_EXPANSIONS)
    parser.add_argument("--max-time-s", type=float, default=DEFAULT_MAX_TIME_S)
    parser.add_argument("--no-lookahead-mask", action="store_true")
    parser.add_argument(
        "--reactive-fallback", action="store_true",
        help="RHCR-style: on a window LaCAM cannot solve in budget, fall back to "
        "greedy one-step-toward-goal per agent instead of freezing the fleet. "
        "Addresses review R2/DA (windowed LaCAM had no fallback).",
    )
    parser.add_argument(
        "--parallel", type=int, default=max(1, (os.cpu_count() or 4) - 2)
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results") / "sprint4b" / "lacam_lifelong",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    rows = run_reference(args)
    summary = summarize(rows, task_budget=args.task_budget)
    write_outputs(rows, summary, args.results_dir)
    print(
        f"\n[lacam_lifelong] tasks_completed = {summary['tasks_completed_mean']} "
        f"+/- {summary['tasks_completed_std']} over {summary['seeds']} seeds"
    )
    if args.task_budget is not None:
        lo, hi = summary["success_wilson95"]  # type: ignore[misc]
        print(
            f"[lacam_lifelong] success regime: budget={args.task_budget} "
            f"-> {summary['success_count']}/{summary['seeds']} succeeded "
            f"(rate {summary['success_rate']}, Wilson95 [{lo}, {hi}])"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
