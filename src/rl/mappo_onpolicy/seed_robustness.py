"""Sprint 5 P1 — Layer-2 router seed robustness for the headroom NO-GO (R1#4).

The headroom NO-GO (greedy ~ Hungarian on throughput) was established on a single
seed-0 Layer-2 router. Reviewer R1#4 asked whether that is an artifact of one
trained router. The curriculum already trained routers at seeds 0/1/2, so this
re-runs the PAIRED greedy-vs-Hungarian throughput comparison on each of them, at
the operating point (lambda=0.1, hotspot + burst_wave), to show the NO-GO holds
across independently trained Layer-2 routers — not just seed 0.

Throughput + paired bootstrap CI only (the clean, unconfounded metric), same as
confirm_headroom: no oracle (degenerate) and no p95 wait (throughput-confounded).
Router seed 0 is included so its numbers reproduce the original headroom probe
(internal-consistency check) and the three routers sit in one comparable table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

_ON_POLICY_ROOT = Path(__file__).resolve().parents[3] / "on-policy"
if str(_ON_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(_ON_POLICY_ROOT))

from src.baselines.hungarian_dispatch import hungarian_dispatch
from src.baselines.stats_utils import paired_comparison
from src.rl.mappo_onpolicy.dispatcher_eval import eval_dispatcher

# Operating-point cells (same lambda as the original headroom probe).
CELLS = [("hotspot", 0.1), ("burst_wave", 0.1)]


def _actor_path(curriculum_dir: Path, dist: str, router_seed: int) -> Path:
    return (
        curriculum_dir / f"seed{router_seed}" / f"n20_{dist}"
        / f"n20_{dist}_seed{router_seed}" / "models" / "actor.pt"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--curriculum-dir", type=Path,
                   default=Path("results") / "sprint4b" / "curriculum")
    p.add_argument("--router-seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="Layer-2 *training* seeds (the external-validity axis)")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)),
                   help="episode/rollout seeds (env stochasticity), paired per cell")
    p.add_argument("--task-rate", type=float, default=0.1)
    p.add_argument("--results-json", type=Path,
                   default=Path("results") / "sprint5" / "seed_robustness.json")
    args = p.parse_args(argv)

    out: dict[str, object] = {
        "episode_n": len(args.seeds),
        "router_seeds": list(args.router_seeds),
        "cells": [],
    }
    for dist, lam in CELLS:
        for rs in args.router_seeds:
            actor = _actor_path(args.curriculum_dir, dist, rs)
            if not actor.exists():
                print(f"[seed_robust] SKIP missing actor: {actor}", flush=True)
                continue
            g = eval_dispatcher(actor, dist, args.seeds, dispatcher=None, task_rate=lam)
            h = eval_dispatcher(actor, dist, args.seeds, dispatcher=hungarian_dispatch,
                                task_rate=lam)
            cmp = paired_comparison(h["throughput"], g["throughput"])  # hung - greedy
            lo, hi = cmp["bootstrap_ci95_mean_diff"]
            sig = (lo > 0) or (hi < 0)
            cell = {
                "distribution": dist, "task_rate": lam, "router_seed": rs,
                "greedy_mean": g["throughput_mean"], "hungarian_mean": h["throughput_mean"],
                "hungarian_minus_greedy": round(cmp["mean_diff"], 2),
                "paired_t_p": round(cmp["paired_t_p"], 4),
                "wilcoxon_p": round(cmp["wilcoxon_p"], 4),
                "bootstrap_ci95": [round(lo, 2), round(hi, 2)],
                "assignment_matters": bool(sig and cmp["mean_diff"] > 0),
            }
            out["cells"].append(cell)
            print(
                f"[seed_robust] {dist} L2-seed{rs} lambda={lam} n={len(args.seeds)}: "
                f"greedy {cell['greedy_mean']} hung {cell['hungarian_mean']} "
                f"(hung-greedy {cell['hungarian_minus_greedy']:+.1f}, p={cell['paired_t_p']}, "
                f"CI {cell['bootstrap_ci95']}) -> assignment matters: {cell['assignment_matters']}",
                flush=True,
            )

    any_matters = any(c["assignment_matters"] for c in out["cells"])
    out["verdict"] = (
        "ASSIGNMENT MATTERS on >=1 router -> reconsider" if any_matters else
        "NO-GO ROBUST ACROSS ROUTER SEEDS: Hungarian never significantly beats greedy"
    )
    print(f"\n[seed_robust] VERDICT: {out['verdict']}", flush=True)

    args.results_json.parent.mkdir(parents=True, exist_ok=True)
    args.results_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[seed_robust] wrote {args.results_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
