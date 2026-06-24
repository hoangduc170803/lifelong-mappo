"""Sprint 4a D4 reward sweep on 5-AGV lifelong uniform.

Runs the 3-way reward comparison from PLAN.md Sprint 4a D4:

    Plan A  (status quo) : progress shaping + r_pickup + r_goal only
    Plan B1 (throughput) : + dense throughput bonus  e * tasks_completed_this_tick
    Plan B2 (idle)       : + idle penalty            -d  per idle agent w/ pending pool
    Plan B1B2 (both)     : B1 + B2 combined

Each (plan x seed) is trained as an isolated subprocess of
``src.rl.mappo_onpolicy.train`` so runs are reproducible and can fan out across
CPU cores. After training, the final ``warehouse/tasks_completed_total`` scalar
is read back from the tensorboardX logs and tabulated.

Decision criterion (PLAN.md D4): pick the reward formulation whose mean
``tasks_completed`` over seeds is >= 0.9 x the PP-32 lifelong reference. The
PP-32 reference is supplied via ``--pp32-reference`` (from D5) or left blank.

Example (full spec):
    python -m src.rl.mappo_onpolicy.reward_sweep \
        --seeds 0 1 2 3 4 --num-env-steps 1000000 --episode-length 1024 \
        --parallel 5 --results-dir results/sprint4a/reward_sweep

Pilot (fast smoke):
    python -m src.rl.mappo_onpolicy.reward_sweep \
        --seeds 0 --num-env-steps 150000 --episode-length 256 --parallel 4 \
        --results-dir results/sprint4a/reward_sweep_pilot
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from src.baselines.stats_utils import paired_comparison
from src.rl.mappo_onpolicy.read_metrics import collect_series

_REPO_ROOT = Path(__file__).resolve().parents[3]
DECISION_TAG = "warehouse/tasks_completed_total"


@dataclass(frozen=True)
class RewardPlan:
    """One reward formulation in the sweep."""

    name: str
    throughput_bonus: float
    idle_penalty: float  # stored as a positive magnitude; applied as negative


def default_plans(throughput_bonus: float, idle_penalty: float) -> list[RewardPlan]:
    return [
        RewardPlan("A_status_quo", 0.0, 0.0),
        RewardPlan("B1_throughput", throughput_bonus, 0.0),
        RewardPlan("B2_idle", 0.0, idle_penalty),
        RewardPlan("B1B2_both", throughput_bonus, idle_penalty),
    ]


@dataclass
class RunResult:
    plan: str
    seed: int
    tasks_completed: float
    elapsed_s: float
    returncode: int
    run_dir: str


def _run_dir(results_dir: Path, plan: str, seed: int) -> Path:
    return results_dir / plan / f"{plan}_seed{seed}"


def _build_argv(
    args: argparse.Namespace,
    plan: RewardPlan,
    seed: int,
) -> list[str]:
    results_dir = _run_dir(args.results_dir, plan.name, seed).parent
    argv = [
        sys.executable,
        "-m",
        "src.rl.mappo_onpolicy.train",
        "--num_agents_target",
        str(args.num_agents),
        "--task_distribution",
        "uniform",
        "--task_rate",
        str(args.task_rate),
        "--episode_length",
        str(args.episode_length),
        "--num_env_steps",
        str(args.num_env_steps),
        "--experiment_name",
        plan.name,
        "--seed",
        str(seed),
        "--results_dir",
        str(results_dir),
        "--n_training_threads",
        str(args.threads_per_run),
        "--log_interval",
        str(args.log_interval),
        "--save_interval",
        "100000",
        "--reward_throughput_bonus_per_task",
        str(plan.throughput_bonus),
        "--reward_idle_when_tasks_pending",
        str(-abs(plan.idle_penalty)),
    ]
    return argv


def _execute(args: argparse.Namespace, plan: RewardPlan, seed: int) -> RunResult:
    run_dir = _run_dir(args.results_dir, plan.name, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    argv = _build_argv(args, plan, seed)
    started = time.perf_counter()
    log_path = run_dir / "train_stdout.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            argv,
            cwd=str(_REPO_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - started
    tasks = _read_tasks_completed(run_dir / "logs")
    return RunResult(
        plan=plan.name,
        seed=seed,
        tasks_completed=tasks,
        elapsed_s=elapsed,
        returncode=proc.returncode,
        run_dir=str(run_dir),
    )


def _read_tasks_completed(log_dir: Path) -> float:
    """Return per-episode throughput averaged over the last training quartile.

    ``warehouse/tasks_completed_total`` is the env's cumulative completed count,
    which resets each episode (auto-reset on horizon). The single last value is
    therefore noisy; averaging the final quartile of logged points gives a far
    more stable estimate of converged throughput.
    """
    if not log_dir.is_dir():
        return float("nan")
    series = collect_series(log_dir)
    s = series.get(DECISION_TAG)
    if s is None or not s.values:
        return float("nan")
    tail = max(1, len(s.values) // 4)
    return float(statistics.mean(s.values[-tail:]))


def run_sweep(args: argparse.Namespace) -> list[RunResult]:
    plans = default_plans(args.throughput_bonus, args.idle_penalty)
    if args.plans:
        wanted = set(args.plans)
        plans = [p for p in plans if p.name in wanted]
        if not plans:
            raise ValueError(f"no plans match {args.plans}")
    jobs = [(plan, seed) for plan in plans for seed in args.seeds]
    args.results_dir.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    workers = max(1, min(args.parallel, len(jobs)))
    print(
        f"[reward_sweep] {len(jobs)} runs "
        f"({len(plans)} plans x {len(args.seeds)} seeds), "
        f"{workers} parallel, {args.num_env_steps} steps each",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_job = {
            pool.submit(_execute, args, plan, seed): (plan, seed)
            for plan, seed in jobs
        }
        done = 0
        for future in concurrent.futures.as_completed(future_to_job):
            result = future.result()
            results.append(result)
            done += 1
            print(
                f"[reward_sweep] {done}/{len(jobs)} {result.plan} "
                f"seed={result.seed} tasks={result.tasks_completed:.1f} "
                f"({result.elapsed_s:.0f}s, rc={result.returncode})",
                flush=True,
            )
    results.sort(key=lambda r: (r.plan, r.seed))
    return results


def summarize(
    results: Sequence[RunResult],
    pp32_reference: Optional[float],
) -> list[dict[str, object]]:
    by_plan: dict[str, list[float]] = {}
    for r in results:
        if r.tasks_completed == r.tasks_completed:  # not NaN
            by_plan.setdefault(r.plan, []).append(r.tasks_completed)
    rows: list[dict[str, object]] = []
    for plan, values in sorted(by_plan.items()):
        mean = statistics.mean(values) if values else float("nan")
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        ratio = (
            mean / pp32_reference
            if pp32_reference not in (None, 0)
            else float("nan")
        )
        passed = (
            ratio >= 0.9
            if pp32_reference not in (None, 0)
            else None
        )
        rows.append(
            {
                "plan": plan,
                "seeds": len(values),
                "tasks_completed_mean": round(mean, 2),
                "tasks_completed_std": round(std, 2),
                "pp32_reference": pp32_reference,
                "ratio_to_pp32": round(ratio, 3) if ratio == ratio else "NA",
                "passes_0.9x": "NA" if passed is None else bool(passed),
            }
        )
    return rows


def paired_stats(
    results: Sequence[RunResult],
    baseline_plan: str = "A_status_quo",
) -> dict[str, object]:
    """Seed-paired significance of each plan vs the status-quo baseline.

    Machine-checkable companion to the markdown: for every non-baseline plan
    sharing seeds with the baseline, emit the paired t-test, Wilcoxon
    signed-rank test, win count, and bootstrap 95% CI of the mean paired diff.
    """
    by_plan: dict[str, dict[int, float]] = {}
    for r in results:
        if r.tasks_completed == r.tasks_completed:  # not NaN
            by_plan.setdefault(r.plan, {})[r.seed] = r.tasks_completed
    base = by_plan.get(baseline_plan, {})
    comparisons: list[dict[str, object]] = []
    for plan, seed_map in sorted(by_plan.items()):
        if plan == baseline_plan:
            continue
        shared = sorted(set(seed_map) & set(base))
        if len(shared) < 2:
            continue
        treatment = [seed_map[s] for s in shared]
        baseline = [base[s] for s in shared]
        comp = {"plan": plan, "baseline": baseline_plan, "seeds": shared}
        comp.update(paired_comparison(treatment, baseline))
        comparisons.append(comp)
    return {"baseline_plan": baseline_plan, "comparisons": comparisons}


def write_outputs(
    results: Sequence[RunResult],
    summary_rows: Sequence[dict[str, object]],
    results_dir: Path,
) -> None:
    runs_csv = results_dir / "reward_sweep_runs.csv"
    with runs_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["plan", "seed", "tasks_completed", "elapsed_s", "returncode", "run_dir"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "plan": r.plan,
                    "seed": r.seed,
                    "tasks_completed": r.tasks_completed,
                    "elapsed_s": round(r.elapsed_s, 1),
                    "returncode": r.returncode,
                    "run_dir": r.run_dir,
                }
            )

    summary_csv = results_dir / "reward_sweep_summary.csv"
    if summary_rows:
        with summary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)

    summary_json = results_dir / "reward_sweep_summary.json"
    summary_json.write_text(
        json.dumps({"summary": list(summary_rows)}, indent=2), encoding="utf-8"
    )

    stats_json = results_dir / "stats_summary.json"
    try:
        stats = paired_stats(results)
        stats_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        wrote_stats = True
    except Exception as exc:  # scipy/numpy missing or degenerate sample
        print(f"[reward_sweep] WARN paired stats skipped: {exc}")
        wrote_stats = False

    print(f"[reward_sweep] wrote {runs_csv}")
    print(f"[reward_sweep] wrote {summary_csv}")
    print(f"[reward_sweep] wrote {summary_json}")
    if wrote_stats:
        print(f"[reward_sweep] wrote {stats_json}")


def _print_table(summary_rows: Sequence[dict[str, object]]) -> None:
    if not summary_rows:
        print("[reward_sweep] no successful runs to summarize")
        return
    print("\n=== Sprint 4a D4 reward sweep (5-AGV lifelong uniform) ===")
    header = f"{'plan':<16} {'seeds':>5} {'tasks_mean':>11} {'tasks_std':>10} {'ratio_pp32':>11} {'pass_0.9x':>10}"
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{row['plan']:<16} {row['seeds']:>5} "
            f"{row['tasks_completed_mean']:>11} {row['tasks_completed_std']:>10} "
            f"{str(row['ratio_to_pp32']):>11} {str(row['passes_0.9x']):>10}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--num-env-steps", type=int, default=1_000_000)
    parser.add_argument("--episode-length", type=int, default=1024)
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--task-rate", type=float, default=0.1)
    parser.add_argument(
        "--throughput-bonus",
        type=float,
        default=0.1,
        help="Plan B1/B1B2 epsilon: reward per task completed this tick.",
    )
    parser.add_argument(
        "--idle-penalty",
        type=float,
        default=0.01,
        help="Plan B2/B1B2 delta magnitude: per-idle-agent penalty when pool nonempty.",
    )
    parser.add_argument(
        "--plans",
        nargs="*",
        default=None,
        help="Subset of plan names to run (default all 4).",
    )
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--threads-per-run", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--pp32-reference",
        type=float,
        default=None,
        help="PP-32 lifelong tasks_completed reference (D5) for the 0.9x criterion.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results") / "sprint4a" / "reward_sweep",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    results = run_sweep(args)
    summary_rows = summarize(results, args.pp32_reference)
    write_outputs(results, summary_rows, args.results_dir)
    _print_table(summary_rows)
    failures = [r for r in results if r.returncode != 0]
    if failures:
        print(f"[reward_sweep] WARNING: {len(failures)} run(s) failed", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
