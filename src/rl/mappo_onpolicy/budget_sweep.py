"""Sprint 4b P1-6 — graded-budget sweep (how far above the floor test?).

GATE-1 condition 3 is a floor test: per-distribution budgets were pinned to
PIBT's pre-freeze capacity (`p2_baseline_matrix.md`), so PASS only certifies
"does not freeze where PIBT does". This sweep re-runs the trained MAPPO policy
at multiples (1x/2x/4x/8x) of the pinned budget to convert the single pass/fail
into a **degrade curve** — answering review R1-S1 ("how far above the floor does
MAPPO sit?"). PIBT cannot reach even 2x by construction: it completes ~pinned
tasks then gridlocks (`p2_baseline_matrix.md`, finding 2), so its success at any
budget > pinned is ~0.

Eval is deterministic, n=30 env-seeds, same env as GATE-1 (λ=0.1, step cap 4096).

    python -m src.rl.mappo_onpolicy.budget_sweep \
        --curriculum-dir results/sprint4b/curriculum --seed 0 \
        --results-json results/sprint4b/budget_sweep/mappo_budget_sweep.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from src.rl.mappo_onpolicy.gate_eval import PINNED, eval_success

DEFAULT_MULTIPLES = (1, 2, 4, 8)


def actor_path(curriculum_dir: Path, seed: int, dist: str) -> Path:
    return (
        curriculum_dir / f"seed{seed}" / f"n20_{dist}"
        / f"n20_{dist}_seed{seed}" / "models" / "actor.pt"
    )


def run_sweep(
    curriculum_dir: Path,
    seed: int,
    dists: Sequence[str],
    multiples: Sequence[int] = DEFAULT_MULTIPLES,
    n_seeds: int = 30,
    encoder: str = "gat",
) -> dict[str, object]:
    cells: list[dict[str, object]] = []
    for dist in dists:
        base = PINNED[dist]["budget"]
        ckpt = actor_path(curriculum_dir, seed, dist)
        for m in multiples:
            budget = base * m
            s = eval_success(
                ckpt, dist, list(range(n_seeds)),
                task_budget=budget, encoder=encoder,
            )
            cells.append({
                "distribution": dist,
                "multiple": m,
                "budget": budget,
                "success_count": s["success_count"],
                "success_rate": s["success_rate"],
                "success_wilson95": s["success_wilson95"],
                "steps_to_budget_mean": s["steps_to_budget_mean"],
                "n": s["n"],
            })
            print(
                f"[budget_sweep] {dist} {m}x (budget {budget}): "
                f"{s['success_count']}/{s['n']} rate {s['success_rate']} "
                f"Wilson {s['success_wilson95']} steps~{s['steps_to_budget_mean']}",
                flush=True,
            )
    return {"seed": seed, "multiples": list(multiples), "cells": cells}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curriculum-dir", type=Path,
        default=Path("results") / "sprint4b" / "curriculum",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--distributions", nargs="+", default=list(PINNED),
        choices=list(PINNED),
    )
    parser.add_argument("--multiples", type=int, nargs="+", default=list(DEFAULT_MULTIPLES))
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--encoder", choices=["mlp", "gat"], default="gat")
    parser.add_argument("--results-json", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out = run_sweep(
        args.curriculum_dir, args.seed, args.distributions,
        multiples=args.multiples, n_seeds=args.n_seeds, encoder=args.encoder,
    )
    if args.results_json is not None:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[budget_sweep] wrote {args.results_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
