"""Sprint 5 — headroom confirm (greedy vs Hungarian; larger-n + higher-lambda).

The headroom probe gave NO-GO at lambda=0.1, n=10 but with two caveats: the burst
Hungarian-over-greedy +16% throughput trend was not significant (n=10), and only
lambda=0.1 (the trained operating point) was tested. This confirms the decision
with PAIRED throughput CI (the clean, unconfounded metric — no oracle, which is
degenerate; no p95 wait, which is throughput-confounded) on:
  - burst_wave lambda=0.1 n=30   (resolve the trend)
  - hotspot    lambda=0.2 n=30   (higher load -> does assignment matter?)
  - burst_wave lambda=0.2 n=30

If Hungarian does NOT significantly beat greedy on any cell, a LEARNED dispatcher
(which must beat Hungarian) has no target -> confirmed NO-GO.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch

_ON_POLICY_ROOT = Path(__file__).resolve().parents[3] / "on-policy"
if str(_ON_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(_ON_POLICY_ROOT))

from src.baselines.hungarian_dispatch import hungarian_dispatch
from src.baselines.stats_utils import paired_comparison
from src.rl.mappo_onpolicy.dispatcher_eval import eval_dispatcher

# (distribution, task_rate) cells; actor is the per-dist seed-0 GAT branch.
CELLS = [
    ("burst_wave", 0.1),
    ("hotspot", 0.2),
    ("burst_wave", 0.2),
]


def _actor_path(curriculum_dir: Path, dist: str) -> Path:
    return (
        curriculum_dir / "seed0" / f"n20_{dist}"
        / f"n20_{dist}_seed0" / "models" / "actor.pt"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--curriculum-dir", type=Path,
                   default=Path("results") / "sprint4b" / "curriculum")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(30)))
    p.add_argument("--results-json", type=Path,
                   default=Path("results") / "sprint5" / "headroom_confirm.json")
    args = p.parse_args(argv)

    out: dict[str, object] = {"n": len(args.seeds), "cells": []}
    for dist, lam in CELLS:
        actor = _actor_path(args.curriculum_dir, dist)
        g = eval_dispatcher(actor, dist, args.seeds, dispatcher=None, task_rate=lam)
        h = eval_dispatcher(actor, dist, args.seeds, dispatcher=hungarian_dispatch,
                            task_rate=lam)
        cmp = paired_comparison(h["throughput"], g["throughput"])  # hung - greedy
        lo, hi = cmp["bootstrap_ci95_mean_diff"]
        sig = (lo > 0) or (hi < 0)
        cell = {
            "distribution": dist, "task_rate": lam,
            "greedy_mean": g["throughput_mean"], "hungarian_mean": h["throughput_mean"],
            "hungarian_minus_greedy": round(cmp["mean_diff"], 2),
            "paired_t_p": round(cmp["paired_t_p"], 4),
            "wilcoxon_p": round(cmp["wilcoxon_p"], 4),
            "bootstrap_ci95": [round(lo, 2), round(hi, 2)],
            "assignment_matters": bool(sig and cmp["mean_diff"] > 0),
        }
        out["cells"].append(cell)
        print(
            f"[confirm] {dist} lambda={lam} n={len(args.seeds)}: "
            f"greedy {cell['greedy_mean']} hung {cell['hungarian_mean']} "
            f"(hung-greedy {cell['hungarian_minus_greedy']:+.1f}, p={cell['paired_t_p']}, "
            f"CI {cell['bootstrap_ci95']}) -> assignment matters: {cell['assignment_matters']}",
            flush=True,
        )

    any_matters = any(c["assignment_matters"] for c in out["cells"])
    out["verdict"] = "ASSIGNMENT MATTERS somewhere -> reconsider" if any_matters else (
        "CONFIRMED NO-GO: Hungarian never significantly beats greedy -> ship MAPPO+Hungarian"
    )
    print(f"\n[confirm] VERDICT: {out['verdict']}", flush=True)

    args.results_json.parent.mkdir(parents=True, exist_ok=True)
    args.results_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[confirm] wrote {args.results_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
