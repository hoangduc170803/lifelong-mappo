"""Sprint 4b — staged curriculum trainer for the warehouse GAT MAPPO router.

Trains a single shared GAT actor through agent-count stages, warm-starting the
(N-invariant) actor across stages while the centralized critic is reinitialized
each stage (its cent_obs dim grows with N). Layout per seed:

    N=5  uniform  ─┐
    N=10 uniform  ─┤ uniform trunk (warm-start actor stage->stage)
    N=15 uniform  ─┘
                   ├─ N=20 hotspot                 ┐
                   ├─ N=20 burst_wave              │ 4 hard-dist branches,
                   ├─ N=20 betweenness_bottleneck  │ each warm-started from
                   └─ N=20 bipartite_pickup_dropoff┘ the N=15 actor

The four N=20 branch policies are the specialists evaluated against the
per-distribution GATE-1 success thresholds pinned in 4b-P2
(0.726 / 0.608 / 0.639 / 0.577 Wilson-upper of PIBT).

Reward plan B1 (throughput bonus eps=0.1/task) is used throughout — the Sprint
4a D4 winner. Training is CPU + parallel-seed fan-out (4b-P4: GPU is
counterproductive at this scale).

Example (real run):
    python -m src.rl.mappo_onpolicy.curriculum --seeds 0 1 2 --parallel 6 \
        --results-dir results/sprint4b/curriculum

Smoke (tiny, validates the warm-start chain end-to-end):
    python -m src.rl.mappo_onpolicy.curriculum --seeds 0 --smoke \
        --results-dir results/sprint4b/curriculum_smoke
"""

from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[3]

HARD_DISTRIBUTIONS = (
    "hotspot",
    "burst_wave",
    "betweenness_bottleneck",
    "bipartite_pickup_dropoff",
)
DEFAULT_THROUGHPUT_BONUS = 0.1  # reward plan B1 (Sprint 4a D4 winner)


@dataclass(frozen=True)
class Stage:
    name: str
    num_agents: int
    distribution: str
    num_env_steps: int
    episode_length: int


def trunk_stages(smoke: bool) -> list[Stage]:
    if smoke:
        return [
            Stage("n5_uniform", 5, "uniform", 256, 64),
            Stage("n10_uniform", 10, "uniform", 256, 64),
            Stage("n15_uniform", 15, "uniform", 256, 64),
        ]
    return [
        Stage("n5_uniform", 5, "uniform", 500_000, 256),
        Stage("n10_uniform", 10, "uniform", 500_000, 256),
        Stage("n15_uniform", 15, "uniform", 1_000_000, 512),
    ]


def branch_stages(smoke: bool) -> list[Stage]:
    steps = 256 if smoke else 2_000_000
    length = 64 if smoke else 1024
    return [
        Stage(f"n20_{d}", 20, d, steps, length) for d in HARD_DISTRIBUTIONS
    ]


def _actor_path(results_dir: Path, seed: int, stage_name: str) -> Path:
    return (
        results_dir
        / f"seed{seed}"
        / stage_name
        / f"{stage_name}_seed{seed}"
        / "models"
        / "actor.pt"
    )


def _stage_results_dir(results_dir: Path, seed: int, stage_name: str) -> Path:
    return results_dir / f"seed{seed}" / stage_name


def _models_dir(results_dir: Path, seed: int, stage_name: str) -> Path:
    return _actor_path(results_dir, seed, stage_name).parent


def stage_status(results_dir: Path, seed: int, stage: Stage) -> tuple[str, int]:
    """Resume helper: ('done'|'partial'|'fresh', remaining_steps).

    Reads the LAST logged ``X/Y`` from train_stdout.log, where Y is the budget
    of the most recent run (the original budget, or a smaller remaining budget
    if the stage was itself resumed — a resumed log holds multiple segments).
    A run is 'done' when its X reached ~its own Y (episode rounding + log
    granularity → 99% margin); otherwise 'partial' with ``Y - X`` left to do.
    'fresh' returns the full stage budget.
    """
    log = _stage_results_dir(results_dir, seed, stage.name) / "train_stdout.log"
    if not log.is_file():
        return "fresh", stage.num_env_steps
    last_x = last_y = 0
    for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "total num timesteps" in line:
            try:
                frac = line.split("total num timesteps")[1].split(",")[0].strip()
                x_str, y_str = frac.split("/")
                last_x, last_y = int(x_str), int(y_str)
            except (IndexError, ValueError):
                continue
    if last_y == 0:
        return "fresh", stage.num_env_steps
    if last_x >= int(0.99 * last_y):
        return "done", 0
    if last_x > 0 and _actor_path(results_dir, seed, stage.name).is_file():
        return "partial", last_y - last_x
    return "fresh", stage.num_env_steps


def _build_argv(
    stage: Stage,
    seed: int,
    results_dir: Path,
    throughput_bonus: float,
    threads_per_run: int,
    actor_init_from: Optional[Path],
    resume_model_dir: Optional[Path] = None,
    steps_override: Optional[int] = None,
    encoder: str = "gat",
) -> list[str]:
    argv = [
        sys.executable,
        "-m",
        "src.rl.mappo_onpolicy.train",
        "--encoder",
        encoder,
        "--num_agents_target",
        str(stage.num_agents),
        "--task_distribution",
        stage.distribution,
        "--episode_length",
        str(stage.episode_length),
        "--num_env_steps",
        str(steps_override if steps_override is not None else stage.num_env_steps),
        "--experiment_name",
        stage.name,
        "--seed",
        str(seed),
        "--results_dir",
        str(_stage_results_dir(results_dir, seed, stage.name)),
        "--n_training_threads",
        str(threads_per_run),
        "--log_interval",
        "5",
        "--save_interval",
        "100000",
        "--reward_throughput_bonus_per_task",
        str(throughput_bonus),
    ]
    if resume_model_dir is not None:
        # Same-N resume: restore BOTH actor and critic, continue remaining steps.
        argv += ["--model_dir", str(resume_model_dir)]
    elif actor_init_from is not None:
        argv += ["--actor_init_from", str(actor_init_from)]
    return argv


@dataclass
class StageResult:
    seed: int
    stage: str
    returncode: int
    elapsed_s: float
    actor_out: str


def _run_stage(
    stage: Stage,
    seed: int,
    results_dir: Path,
    throughput_bonus: float,
    threads_per_run: int,
    actor_init_from: Optional[Path],
    resume: bool = False,
    encoder: str = "gat",
) -> StageResult:
    run_dir = _stage_results_dir(results_dir, seed, stage.name)
    run_dir.mkdir(parents=True, exist_ok=True)

    resume_model_dir: Optional[Path] = None
    steps_override: Optional[int] = None
    log_mode = "w"
    if resume:
        status, remaining = stage_status(results_dir, seed, stage)
        if status == "done":
            return StageResult(seed, stage.name, 0, 0.0,
                               str(_actor_path(results_dir, seed, stage.name)))
        if status == "partial":
            # Restore actor+critic from this stage's last checkpoint and train
            # only the remaining steps. The on-policy step counter restarts, so
            # at most one save_interval (100k) of progress is re-done.
            resume_model_dir = _models_dir(results_dir, seed, stage.name)
            steps_override = remaining
            log_mode = "a"

    argv = _build_argv(
        stage, seed, results_dir, throughput_bonus, threads_per_run,
        actor_init_from, resume_model_dir=resume_model_dir,
        steps_override=steps_override, encoder=encoder,
    )
    started = time.perf_counter()
    log_path = run_dir / "train_stdout.log"
    with log_path.open(log_mode, encoding="utf-8") as log_file:
        proc = subprocess.run(
            argv,
            cwd=str(_REPO_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - started
    return StageResult(
        seed=seed,
        stage=stage.name,
        returncode=proc.returncode,
        elapsed_s=elapsed,
        actor_out=str(_actor_path(results_dir, seed, stage.name)),
    )


def _plan_branch_jobs(
    seeds: Sequence[int],
    branches: Sequence[Stage],
    n15_actor: dict[int, Optional[Path]],
    skip_trunk: bool,
) -> list[tuple[Stage, int, Optional[Path]]]:
    """Build the (stage, seed, warm-start-init) branch jobs.

    A ``None`` entry in ``n15_actor`` is overloaded: under ``--skip-trunk`` it
    legitimately means "train this branch from scratch" and is kept; in normal
    curriculum mode it means the seed's trunk FAILED, so that seed's branches
    are dropped instead of spawning warm-start jobs against a dead checkpoint.
    """
    jobs: list[tuple[Stage, int, Optional[Path]]] = []
    for seed in seeds:
        init = n15_actor.get(seed)
        if not skip_trunk and init is None:
            continue  # trunk failed for this seed -> skip its branches
        for stage in branches:
            jobs.append((stage, seed, init))
    return jobs


def run_curriculum(args: argparse.Namespace) -> list[StageResult]:
    trunk = trunk_stages(args.smoke)
    branches = branch_stages(args.smoke)
    results: list[StageResult] = []
    args.results_dir.mkdir(parents=True, exist_ok=True)

    encoder = getattr(args, "encoder", "gat")
    skip_trunk = getattr(args, "skip_trunk", False)

    # --- Trunk: each seed runs the uniform stages sequentially (warm-start
    # chained); different seeds fan out in parallel. ---
    def run_trunk(seed: int) -> Optional[Path]:
        prev_actor: Optional[Path] = None
        for stage in trunk:
            res = _run_stage(
                stage,
                seed,
                args.results_dir,
                args.throughput_bonus,
                args.threads_per_run,
                prev_actor,
                resume=args.resume,
                encoder=encoder,
            )
            results.append(res)
            print(
                f"[curriculum] seed={seed} {stage.name} done "
                f"({res.elapsed_s:.0f}s, rc={res.returncode})",
                flush=True,
            )
            if res.returncode != 0:
                print(f"[curriculum] seed={seed} ABORT trunk at {stage.name}")
                return None
            prev_actor = _actor_path(args.results_dir, seed, stage.name)
        # All trunk stages reported rc=0; the final N=15 actor must actually
        # exist to warm-start the branches. If it is missing (rc=0 but no save),
        # treat the trunk as failed rather than handing a dead path downstream.
        if prev_actor is None or not prev_actor.exists():
            print(
                f"[curriculum] seed={seed} trunk done but N=15 actor missing "
                f"({prev_actor}) -> marking trunk failed"
            )
            # Record an explicit failure so main()'s returncode filter trips
            # (exit 1). Without this, an rc=0-but-no-checkpoint trunk would skip
            # the seed's branches silently and the run would still exit green.
            results.append(
                StageResult(
                    seed=seed,
                    stage=f"{trunk[-1].name}(actor_missing)",
                    returncode=-1,
                    elapsed_s=0.0,
                    actor_out=str(prev_actor) if prev_actor is not None else "",
                )
            )
            return None
        return prev_actor  # N=15 actor

    workers = max(1, args.parallel)
    n15_actor: dict[int, Optional[Path]] = {}
    if skip_trunk:
        # Curriculum ablation: train N=20 branches directly from scratch (no
        # 5->10->15 warm-start). Each branch initializes fresh.
        for s in args.seeds:
            n15_actor[s] = None
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            fut = {pool.submit(run_trunk, s): s for s in args.seeds}
            for f in concurrent.futures.as_completed(fut):
                n15_actor[fut[f]] = f.result()

    # --- Branches: 4 hard-dist N=20 specialists per seed, each warm-started
    # from that seed's N=15 actor. All (seed x branch) fan out in parallel. ---
    if not skip_trunk:
        for s in args.seeds:
            if n15_actor.get(s) is None:
                print(f"[curriculum] seed={s} SKIP branches (trunk failed)")
    branch_jobs = _plan_branch_jobs(args.seeds, branches, n15_actor, skip_trunk)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        fut = {
            pool.submit(
                _run_stage,
                stage,
                seed,
                args.results_dir,
                args.throughput_bonus,
                args.threads_per_run,
                init,
                args.resume,
                encoder,
            ): (stage.name, seed)
            for stage, seed, init in branch_jobs
        }
        for f in concurrent.futures.as_completed(fut):
            res = f.result()
            results.append(res)
            print(
                f"[curriculum] seed={res.seed} {res.stage} done "
                f"({res.elapsed_s:.0f}s, rc={res.returncode})",
                flush=True,
            )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--parallel", type=int, default=6)
    parser.add_argument("--threads-per-run", type=int, default=2)
    parser.add_argument(
        "--throughput-bonus", type=float, default=DEFAULT_THROUGHPUT_BONUS
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--encoder", choices=["mlp", "gat"], default="gat",
        help="Actor encoder for all stages (P1 encoder ablation: mlp).",
    )
    parser.add_argument(
        "--skip-trunk", action="store_true",
        help="P1 curriculum ablation: train N=20 branches direct from scratch "
             "(no 5->10->15 warm-start).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted run: skip stages whose log already reached "
            "the step budget, restore in-progress stages from their last "
            "checkpoint (actor+critic) and train only the remaining steps. "
            "Up to one save_interval (100k steps) of an in-progress stage is "
            "re-done (on-policy does not persist optimizer/step state)."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results") / "sprint4b" / "curriculum",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    results = run_curriculum(args)
    failed = [r for r in results if r.returncode != 0]
    print(
        f"\n[curriculum] {len(results)} stage-runs in "
        f"{(time.perf_counter() - started) / 3600:.2f} h; "
        f"{len(failed)} failed"
    )
    for r in failed:
        print(f"  FAILED seed={r.seed} {r.stage} rc={r.returncode}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
