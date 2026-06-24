"""Sprint 4b P3 — PIBT & LaCAM one-shot stress reference on the warehouse graph.

GATE 1 condition (2) carries the clause "MAPPO must beat PIBT and LaCAM on at
least one of the same hard distributions" (plan review R3 — "PP-32 fails ≠ all
classical fail"). This runner produces that classical bar in the **one-shot**
regime, the complement of the P2 lifelong matrix:

- same 4 hard distributions as P2, sampled as ONE-SHOT MAPF instances via
  `sprint35_gate.sample_warehouse_stress_instance` (distinct starts, clustered
  goals); N=20; n=30 seeds.
- **PIBT** (Okumura 2022): iterated one-step planning to completion — replan
  one joint move per tick, dynamic priorities (grow while an agent's goal is
  unreached, reset on arrival), until every agent is at its goal or a step cap
  is hit. Success = all-at-goal within cap.
- **LaCAM** (Okumura 2023): a single `plan()` call; success = a conflict-free
  joint plan was returned within the expansion/time budget.

Both run on the raw directed graph WITHOUT the env's lookahead action mask:
this is the planners' native regime (one-shot MAPF), and the point of the
contrast with P2 is exactly that classical planners *do* solve isolated
instances — the lifelong collapse is a property of sustained operation, not of
the planners' raw competence.

Example:
    python -m src.baselines.p3_classical_stress \
        --seeds 0..29 --num-agents 20 --parallel 10 \
        --results-dir results/sprint4b/p3_classical_stress_n20
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

import networkx as nx

from src.baselines.benchmark import DEFAULT_MAP_FILE
from src.baselines.stats_utils import wilson_ci
from src.baselines.sprint35_gate import (
    STRESS_DISTRIBUTIONS,
    sample_warehouse_stress_instance,
)
from src.map_parser import parse_opentcs_map
from src.mapf.cbs_solver import validate_paths
from src.mapf.lacam import LaCAMPlanner
from src.mapf.pibt import PIBT
from src.routing.astar import AStarRouter

# A one-shot instance needs at most a few times the longest single-agent path
# to clear; the warehouse SCC diameter is ~300 steps, so 4x bounds a healthy
# completion window without letting a livelocked PIBT run forever.
DEFAULT_STEP_CAP = 1024
DEFAULT_LACAM_MAX_EXPANSIONS = 30_000
DEFAULT_LACAM_MAX_TIME_S = 30.0


def run_pibt_oneshot(
    G: nx.DiGraph,
    dist,
    starts: dict[str, str],
    goals: dict[str, str],
    seed: int,
    step_cap: int = DEFAULT_STEP_CAP,
) -> dict[str, object]:
    """Iterate PIBT one joint move per tick until all-at-goal or step cap."""
    planner = PIBT(G, dist, seed=seed)
    positions = dict(starts)
    priorities: dict[str, float] = {a: 0.0 for a in starts}
    paths: dict[str, list[str]] = {a: [positions[a]] for a in starts}
    started = time.perf_counter()
    steps = 0
    success = False
    for _ in range(step_cap):
        if all(positions[a] == goals[a] for a in positions):
            success = True
            break
        for a in positions:
            priorities[a] = (
                priorities[a] + 1.0 if positions[a] != goals[a] else 0.0
            )
        moves = planner.step(positions, goals, priorities)
        if moves is None:
            break
        positions = moves
        steps += 1
        for a in positions:
            paths[a].append(positions[a])
    if all(positions[a] == goals[a] for a in positions):
        success = True
    elapsed = time.perf_counter() - started
    conflicts = validate_paths(G, paths) if success else []
    return {
        "baseline": "pibt_oneshot",
        "success": bool(success and not conflicts),
        "makespan": steps if success else 0,
        "conflicts": len(conflicts),
        "elapsed_s": round(elapsed, 4),
    }


def run_lacam_oneshot(
    G: nx.DiGraph,
    dist,
    starts: dict[str, str],
    goals: dict[str, str],
    seed: int,
    max_expansions: int = DEFAULT_LACAM_MAX_EXPANSIONS,
    max_time_s: float = DEFAULT_LACAM_MAX_TIME_S,
) -> dict[str, object]:
    planner = LaCAMPlanner(
        G, dist, seed=seed, max_expansions=max_expansions, max_time_s=max_time_s
    )
    started = time.perf_counter()
    result = planner.plan(starts, goals)
    elapsed = time.perf_counter() - started
    conflicts = validate_paths(G, result.paths) if result.success else []
    return {
        "baseline": "lacam_oneshot",
        "success": bool(result.success and not conflicts),
        "makespan": result.makespan if result.success else 0,
        "conflicts": len(conflicts),
        "expansions": int(result.diagnostics.get("expansions", 0)),
        "elapsed_s": round(elapsed, 4),
    }


def run_instance(
    G: nx.DiGraph,
    router: AStarRouter,
    distribution: str,
    num_agents: int,
    seed: int,
    step_cap: int,
    lacam_max_expansions: int,
    lacam_max_time_s: float,
) -> list[dict[str, object]]:
    starts, goals = sample_warehouse_stress_instance(
        G, num_agents, seed, distribution
    )
    rows: list[dict[str, object]] = []
    for runner in (
        run_pibt_oneshot(
            G, router.distance, starts, goals, seed, step_cap=step_cap
        ),
        run_lacam_oneshot(
            G,
            router.distance,
            starts,
            goals,
            seed,
            max_expansions=lacam_max_expansions,
            max_time_s=lacam_max_time_s,
        ),
    ):
        runner.update(
            {"distribution": distribution, "num_agents": num_agents, "seed": seed}
        )
        rows.append(runner)
    return rows


# ---------------------------------------------------------------------- runner

_WORKER_G: Optional[nx.DiGraph] = None
_WORKER_ROUTER: Optional[AStarRouter] = None


def _init_worker(map_file: str) -> None:
    global _WORKER_G, _WORKER_ROUTER
    _WORKER_G = parse_opentcs_map(map_file, restrict_to_largest_scc=True)
    _WORKER_ROUTER = AStarRouter(_WORKER_G, precompute=True)


def _run_job(job: tuple) -> list[dict[str, object]]:
    if _WORKER_G is None or _WORKER_ROUTER is None:
        _init_worker(str(DEFAULT_MAP_FILE))
    assert _WORKER_G is not None and _WORKER_ROUTER is not None
    return run_instance(_WORKER_G, _WORKER_ROUTER, *job)


def run_reference(args: argparse.Namespace) -> list[dict[str, object]]:
    jobspecs = [
        (
            dist,
            args.num_agents,
            seed,
            args.step_cap,
            args.lacam_max_expansions,
            args.lacam_max_time_s,
        )
        for dist in args.distributions
        for seed in args.seeds
    ]
    rows: list[dict[str, object]] = []
    workers = max(1, min(args.parallel, len(jobspecs)))
    if workers == 1:
        G = parse_opentcs_map(str(DEFAULT_MAP_FILE), restrict_to_largest_scc=True)
        router = AStarRouter(G, precompute=True)
        for i, job in enumerate(jobspecs, start=1):
            rows.extend(run_instance(G, router, *job))
            print(f"[p3] {i}/{len(jobspecs)} {job[0]} seed={job[2]}", flush=True)
        return rows

    print(f"[p3] {len(jobspecs)} instances across {workers} processes", flush=True)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(DEFAULT_MAP_FILE),),
    ) as pool:
        future_to_job = {pool.submit(_run_job, job): job for job in jobspecs}
        done = 0
        for future in concurrent.futures.as_completed(future_to_job):
            rows.extend(future.result())
            done += 1
            job = future_to_job[future]
            print(f"[p3] {done}/{len(jobspecs)} {job[0]} seed={job[2]}", flush=True)
    rows.sort(
        key=lambda r: (str(r["distribution"]), int(r["seed"]), str(r["baseline"]))
    )
    return rows


def summarize(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    """Per-(baseline, distribution) success + makespan with Wilson CI."""
    cells: dict[tuple[str, str], list[dict[str, object]]] = {}
    for r in rows:
        cells.setdefault((str(r["baseline"]), str(r["distribution"])), []).append(r)
    summary: dict[str, object] = {"cells": []}
    for (baseline, dist), cell in sorted(cells.items()):
        n = len(cell)
        succ = sum(int(r["success"]) for r in cell)
        lo, hi = wilson_ci(succ, n)
        mk = [int(r["makespan"]) for r in cell if r["success"]]
        el = [float(r["elapsed_s"]) for r in cell]
        summary["cells"].append(
            {
                "baseline": baseline,
                "distribution": dist,
                "n": n,
                "success_count": succ,
                "success_rate": round(succ / n, 3),
                "success_wilson95": [round(lo, 3), round(hi, 3)],
                "makespan_mean": round(statistics.mean(mk), 1) if mk else None,
                "makespan_max": max(mk) if mk else None,
                "elapsed_s_mean": round(statistics.mean(el), 4),
                "elapsed_s_max": round(max(el), 4),
            }
        )
    return summary


def write_outputs(
    rows: Sequence[dict[str, object]],
    summary: dict[str, object],
    results_dir: Path,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "baseline",
        "distribution",
        "num_agents",
        "seed",
        "success",
        "makespan",
        "conflicts",
        "expansions",
        "elapsed_s",
    ]
    runs_csv = results_dir / "p3_classical_stress_runs.csv"
    with runs_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary_json = results_dir / "p3_classical_stress_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[p3] wrote {runs_csv}")
    print(f"[p3] wrote {summary_json}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(30)))
    parser.add_argument("--num-agents", type=int, default=20)
    parser.add_argument(
        "--distributions", nargs="+", default=list(STRESS_DISTRIBUTIONS)
    )
    parser.add_argument("--step-cap", type=int, default=DEFAULT_STEP_CAP)
    parser.add_argument(
        "--lacam-max-expansions", type=int, default=DEFAULT_LACAM_MAX_EXPANSIONS
    )
    parser.add_argument(
        "--lacam-max-time-s", type=float, default=DEFAULT_LACAM_MAX_TIME_S
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 2),
        help="Worker processes for instance fan-out (default: cpu_count - 2).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results") / "sprint4b" / "p3_classical_stress_n20",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    rows = run_reference(args)
    summary = summarize(rows)
    write_outputs(rows, summary, args.results_dir)
    print("\n[p3] one-shot classical stress (N={}):".format(args.num_agents))
    for cell in summary["cells"]:  # type: ignore[index]
        lo, hi = cell["success_wilson95"]
        print(
            f"  {cell['baseline']:14s} {cell['distribution']:24s} "
            f"success {cell['success_count']}/{cell['n']} "
            f"(Wilson [{lo}, {hi}]) makespan~{cell['makespan_mean']} "
            f"t~{cell['elapsed_s_mean']}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
