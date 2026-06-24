"""Sprint 4a D6 — PIBT lifelong reference on the warehouse env.

Drives the SAME lifelong ``WarehouseEnv`` used for MAPPO training (greedy
dispatch, Poisson arrivals, pool cap, safety validator, lookahead action
mask, compass actions) with PIBT (Okumura 2022) replacing the learned
policy. Sibling of `src/baselines/pp32_lifelong.py`; produces the second
classical reference for the D4/GATE-1 comparison table.

Control loop: PIBT plans ONE joint move per tick from the env's dispatch
goals. Candidates are restricted to action-mask-valid nodes (the same mask
MAPPO observes), so PIBT plays by identical env rules. Dynamic priorities
follow the paper's lifelong scheme: an agent's priority grows each tick it
has a task it has not yet reached, and resets on arrival (ties broken by
agent id inside the planner). The ``--priority-scheme`` switch (R2 ablation)
swaps this for random / distance-to-goal / static-index orderings; the N=20
freeze is robust to all four (results/sprint6/pibt_priority_ablation.md).

Example (full reference):
    python -m src.baselines.pibt_lifelong \
        --seeds 0..29 --episode-length 1024 --parallel 10 \
        --results-dir results/sprint4a/pibt_lifelong
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import random
import statistics
import time
from pathlib import Path
from typing import Optional, Sequence

from src.baselines.stats_utils import wilson_ci
from src.env.compass_mapper import WAIT_SLOT
from src.env.warehouse_env import WarehouseEnv
from src.mapf.pibt import PIBT
from src.rl.mappo_onpolicy.env_adapter import (
    WarehouseEnvConfig,
    build_warehouse_env,
)


class PIBTLifelongController:
    """Action provider: one PIBT joint move per tick -> compass slots.

    Carries the same gridlock-recovery heuristic as the PP-32 wrapper
    (`pp32_lifelong.PP32LifelongController`): at N=20 the agents converge
    on the task hotspots and lock into a mutual-occupancy blob — interior
    agents have no open exit, boundary agents have one but never take it
    because staying (closer to goal) dominates a sidestep, so the whole
    fleet freezes forever (seen on every hotspot/burst seed of the first
    4096-step run: last task completion at step ~60, zero movement after).
    PIBT's priority inheritance cannot help: the mask has already removed
    the contended candidates. Recovery mirrors PP-32's: any agent with an
    unreached goal that has not MOVED for `freeze_patience` consecutive
    ticks becomes "stuck"; one stuck agent per tick (lowest name,
    deterministic) is forced onto any mask-open non-WAIT slot, dissolving
    the blob from its boundary.
    """

    def __init__(
        self,
        env: WarehouseEnv,
        seed: int = 0,
        freeze_patience: int = 5,
        priority_scheme: str = "elapsed",
    ):
        self.env = env
        self.planner = PIBT(env.G, env.router.distance, seed=seed)
        self.priority_scheme = priority_scheme
        self._prio_rng = random.Random(seed + 104729)  # independent of planner RNG
        self._priorities: dict[str, float] = {}
        self.freeze_patience = freeze_patience
        self._static_ticks: dict[str, int] = {}
        self._last_pos: dict[str, str] = {}
        self.freeze_sidesteps = 0

    def actions(self) -> dict[str, int]:
        agents = self.env.agents
        positions: dict[str, str] = {}
        goals: dict[str, str] = {}
        for a in agents:
            info = self.env._agent_info[a]
            positions[a] = info.pos
            goals[a] = info.task.current_goal if info.task is not None else info.pos

        # Per-agent static streaks (positions are compared across ticks, so
        # this sees the env's ACTUAL outcome, validator forced-waits included).
        for a in agents:
            if positions[a] == self._last_pos.get(a) and goals[a] != positions[a]:
                self._static_ticks[a] = self._static_ticks.get(a, 0) + 1
            else:
                self._static_ticks[a] = 0
        self._last_pos = dict(positions)

        # Same per-agent action mask MAPPO receives in its observation:
        # restrict PIBT's candidates to env-legal moves.
        masks, slot_maps, _ = self.env._action_masks_for_agents(agents)
        allowed = {
            a: {
                node
                for slot, node in slot_maps[a].items()
                if slot != WAIT_SLOT and masks[a][slot]
            }
            for a in agents
        }

        # Dynamic priorities. Default lifelong scheme (paper SV): grow each tick
        # a task goal is unreached, reset on arrival. The priority_scheme switch
        # (R2 priority-config ablation) swaps in alternative orderings to show the
        # N=20 freeze is set by congestion + the action mask, not by this one
        # configuration. A reached goal is always priority 0 in every scheme.
        for a in agents:
            if goals[a] == positions[a]:
                self._priorities[a] = 0.0
            elif self.priority_scheme == "elapsed":
                self._priorities[a] = self._priorities.get(a, 0.0) + 1.0
            elif self.priority_scheme == "random":
                self._priorities[a] = self._prio_rng.random()
            elif self.priority_scheme == "distance":
                self._priorities[a] = float(self.planner.dist(positions[a], goals[a]))
            elif self.priority_scheme == "index":
                self._priorities[a] = 0.0  # all equal -> planner tie-breaks by id
            else:
                raise ValueError(
                    f"unknown priority_scheme: {self.priority_scheme!r}"
                )

        moves = self.planner.step(positions, goals, self._priorities, allowed)

        acts: dict[str, int] = {}
        for a in agents:
            target = moves[a]
            if target == positions[a]:
                acts[a] = WAIT_SLOT
                continue
            slot = self._slot_to(positions[a], target)
            acts[a] = WAIT_SLOT if slot is None else slot

        stuck = sorted(
            a
            for a in agents
            if self._static_ticks.get(a, 0) >= self.freeze_patience
            and acts[a] == WAIT_SLOT
        )
        if stuck:
            agent = stuck[0]
            open_slots = [
                slot
                for slot, node in slot_maps[agent].items()
                if slot != WAIT_SLOT and masks[agent][slot]
            ]
            if open_slots:
                acts[agent] = int(open_slots[0])
                self.freeze_sidesteps += 1
                self._static_ticks[agent] = 0
        return acts

    def _slot_to(self, src: str, target: str) -> Optional[int]:
        _, slot_to_node = self.env.compass.get(src)
        for slot, node in slot_to_node.items():
            if node == target:
                return int(slot)
        return None


# ---------------------------------------------------------------------- runner


def run_episode(
    env: WarehouseEnv, seed: int, priority_scheme: str = "elapsed"
) -> dict[str, object]:
    env.reset(seed=seed)
    controller = PIBTLifelongController(
        env, seed=seed, priority_scheme=priority_scheme
    )
    started = time.perf_counter()
    tasks_completed = 0
    validator_interventions = 0
    budget_reached = False
    steps = 0
    while env.agents:
        acts = controller.actions()
        _, _, terminated, truncated, infos = env.step(acts)
        steps += 1
        sample = infos[next(iter(infos))]
        tasks_completed = int(sample["tasks_completed_total"])
        validator_interventions += int(sample["validator_interventions"])
        budget_reached = bool(sample["task_budget_reached"])
        if all(terminated.values()) or all(truncated.values()):
            break
    elapsed = time.perf_counter() - started
    return {
        "baseline": "pibt_lifelong",
        "priority_scheme": controller.priority_scheme,
        "seed": seed,
        "steps": steps,
        "tasks_completed": tasks_completed,
        "task_budget_reached": int(budget_reached),
        "throughput_per_step": round(tasks_completed / max(steps, 1), 4),
        "freeze_sidesteps": controller.freeze_sidesteps,
        "validator_interventions": validator_interventions,
        "wall_s": round(elapsed, 1),
    }


_WORKER_ENV: Optional[WarehouseEnv] = None
_WORKER_CFG: Optional[WarehouseEnvConfig] = None


def _run_seed_job(
    cfg: WarehouseEnvConfig, seed: int, priority_scheme: str = "elapsed"
) -> dict[str, object]:
    global _WORKER_ENV, _WORKER_CFG
    if _WORKER_ENV is None or _WORKER_CFG != cfg:
        _WORKER_ENV = build_warehouse_env(cfg)
        _WORKER_CFG = cfg
    return run_episode(_WORKER_ENV, seed=seed, priority_scheme=priority_scheme)


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
            row = run_episode(env, seed=seed, priority_scheme=args.priority_scheme)
            rows.append(row)
            print(
                f"[pibt_lifelong] {i}/{len(args.seeds)} seed={seed} "
                f"tasks={row['tasks_completed']} ({row['wall_s']}s)",
                flush=True,
            )
        return rows

    print(
        f"[pibt_lifelong] {len(args.seeds)} seeds across {workers} processes",
        flush=True,
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_seed = {
            pool.submit(_run_seed_job, cfg, seed, args.priority_scheme): seed
            for seed in args.seeds
        }
        done = 0
        for future in concurrent.futures.as_completed(future_to_seed):
            row = future.result()
            rows.append(row)
            done += 1
            print(
                f"[pibt_lifelong] {done}/{len(args.seeds)} seed={row['seed']} "
                f"tasks={row['tasks_completed']} ({row['wall_s']}s)",
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
        "baseline": "pibt_lifelong",
        "seeds": len(tasks),
        "tasks_completed_mean": round(mean, 2),
        "tasks_completed_std": round(std, 2),
        "tasks_completed_min": min(tasks) if tasks else None,
        "tasks_completed_max": max(tasks) if tasks else None,
        "d4_criterion_0.9x": round(0.9 * mean, 2),
    }
    if task_budget is not None:
        n = len(rows)
        successes = sum(int(r["task_budget_reached"]) for r in rows)
        lo, hi = wilson_ci(successes, n)
        steps_succ = [
            int(r["steps"]) for r in rows if int(r["task_budget_reached"])
        ]
        summary.update(
            {
                "task_budget": task_budget,
                "success_count": successes,
                "success_rate": round(successes / n, 3),
                "success_wilson95": [round(lo, 3), round(hi, 3)],
                "steps_to_budget_mean": (
                    round(statistics.mean(steps_succ), 1) if steps_succ else None
                ),
                "steps_to_budget_max": max(steps_succ) if steps_succ else None,
            }
        )
    return summary


def write_outputs(
    rows: Sequence[dict[str, object]],
    summary: dict[str, object],
    results_dir: Path,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = results_dir / "pibt_lifelong_runs.csv"
    with runs_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_json = results_dir / "pibt_lifelong_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[pibt_lifelong] wrote {runs_csv}")
    print(f"[pibt_lifelong] wrote {summary_json}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(30)))
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--episode-length", type=int, default=1024)
    parser.add_argument("--task-rate", type=float, default=0.1)
    parser.add_argument("--task-distribution", default="uniform")
    parser.add_argument(
        "--priority-scheme",
        choices=["elapsed", "random", "distance", "index"],
        default="elapsed",
        help=(
            "Dynamic-priority ordering for PIBT (paper SV default = 'elapsed', "
            "steps since last goal). 'random'/'distance'/'index' are the R2 "
            "priority-config ablation: the N=20 freeze persists under all four."
        ),
    )
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
    parser.add_argument(
        "--parallel",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 2),
        help="Worker processes for seed fan-out (default: cpu_count - 2).",
    )
    parser.add_argument(
        "--no-lookahead-mask",
        action="store_true",
        help="Disable the env's lookahead action mask (control condition).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results") / "sprint4a" / "pibt_lifelong",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    rows = run_reference(args)
    summary = summarize(rows, task_budget=args.task_budget)
    summary["priority_scheme"] = args.priority_scheme
    write_outputs(rows, summary, args.results_dir)
    print(
        f"\n[pibt_lifelong] reference: tasks_completed = "
        f"{summary['tasks_completed_mean']} +/- {summary['tasks_completed_std']} "
        f"over {summary['seeds']} seeds"
    )
    if args.task_budget is not None:
        lo, hi = summary["success_wilson95"]  # type: ignore[misc]
        print(
            f"[pibt_lifelong] success regime: budget={args.task_budget} "
            f"-> {summary['success_count']}/{summary['seeds']} succeeded "
            f"(rate {summary['success_rate']}, Wilson95 [{lo}, {hi}])"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
