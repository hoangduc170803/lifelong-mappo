"""Sprint 4b P1-5 — load sweep: throughput vs task arrival rate λ.

The GATE-1 gate is a floor test and throughput at λ=0.1 sits near the arrival
ceiling, so the ablations could not separate GAT from MLP (p1_ablations.md).
This sweep lifts the ceiling: it measures **throughput** (tasks completed per
1024-step episode, no task_budget terminal) for a trained policy across
λ ∈ {0.1, 0.2, 0.4}. Comparing GAT vs MLP under heavier load tests whether the
GAT encoder earns its keep; comparing learned vs classical (run separately via
`pibt_lifelong`/`lacam_lifelong --task-rate`) shows whether the gap is graded.

NOTE: a policy trained at λ=0.1 is evaluated out-of-distribution at higher λ;
both encoders are equally OOD, so the GAT-vs-MLP contrast stays fair, but
absolute numbers at λ>0.1 are generalization, not trained performance.

Example:
    python -m src.rl.mappo_onpolicy.load_sweep \
        --actor .../n20_hotspot_seed0/models/actor.pt --encoder gat \
        --distribution hotspot --task-rate 0.2 --seeds 0..29
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

_ON_POLICY_ROOT = Path(__file__).resolve().parents[3] / "on-policy"
if str(_ON_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(_ON_POLICY_ROOT))

from src.baselines.stats_utils import welch_comparison
from src.rl.mappo_onpolicy.env_adapter import WarehouseEnvConfig, WarehouseOnPolicyEnv
from src.rl.mappo_onpolicy.gate_eval import _eval_args, load_actor

DEFAULT_HORIZON = 1024


def eval_throughput(
    checkpoint: Path,
    distribution: str,
    seeds: Sequence[int],
    task_rate: float,
    num_agents: int = 20,
    horizon: int = DEFAULT_HORIZON,
    encoder: str = "gat",
    device: Optional[torch.device] = None,
) -> dict[str, object]:
    device = device or torch.device("cpu")
    args = _eval_args(num_agents, encoder=encoder)
    cfg = WarehouseEnvConfig(
        num_agents=num_agents,
        episode_horizon=horizon,
        task_rate=task_rate,
        task_distribution=distribution,
        knn_agents=args.knn_agents,  # no task_budget -> throughput regime
    )
    wrap = WarehouseOnPolicyEnv(cfg, auto_reset=False)
    actor = load_actor(
        checkpoint, args, wrap.observation_space[0], wrap.action_space[0], device
    )
    hidden, rec_n = args.hidden_size, args.recurrent_N
    tasks: list[int] = []
    for seed in seeds:
        obs, _, avail = wrap.reset(seed=seed)
        rnn = np.zeros((num_agents, rec_n, hidden), dtype=np.float32)
        masks = np.ones((num_agents, 1), dtype=np.float32)
        completed = 0
        for _ in range(horizon):
            with torch.no_grad():
                actions, _, _ = actor(obs, rnn, masks, avail, deterministic=True)
            obs, _, _, dones, info, avail = wrap.step(
                actions.detach().cpu().numpy()
            )
            completed = int(info["tasks_completed_total"])
            if bool(np.asarray(dones).all()):
                break
        tasks.append(completed)
    wrap.close()
    return {
        "encoder": encoder,
        "distribution": distribution,
        "task_rate": task_rate,
        "n": len(tasks),
        "throughput_mean": round(statistics.mean(tasks), 2),
        "throughput_std": round(statistics.pstdev(tasks) if len(tasks) > 1 else 0.0, 2),
        "tasks": tasks,
    }


def aggregate(results_dir: Path) -> dict[str, object]:
    """Pair the per-run ``{enc}_{dist}_l{rate}.json`` files and emit GAT-vs-MLP
    Welch tests + the PIBT classical reference per (distribution, lambda).

    Machine-checkable companion to ``p1_load_sweep.md``: the p-values that the
    markdown quotes are recomputed here from the saved per-seed ``tasks`` lists
    so the thesis audit does not depend on prose.
    """
    def _load(enc: str, dist: str, rate: float) -> Optional[dict]:
        path = results_dir / f"{enc}_{dist}_l{rate}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _pibt(dist: str, rate: float) -> Optional[float]:
        path = results_dir / f"pibt_{dist}_l{rate}" / "pibt_lifelong_summary.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in ("throughput_mean", "tasks_completed_mean", "mean_tasks_completed"):
            if key in data:
                return float(data[key])
        return None

    gat_files = sorted(results_dir.glob("gat_*_l*.json"))
    cells: list[dict[str, object]] = []
    for gpath in gat_files:
        stem = gpath.stem[len("gat_"):]  # "<dist>_l<rate>"
        dist, _, rate_s = stem.rpartition("_l")
        rate = float(rate_s)
        gat = _load("gat", dist, rate)
        mlp = _load("mlp", dist, rate)
        if gat is None or mlp is None:
            continue
        cell = {"distribution": dist, "task_rate": rate}
        cell.update(welch_comparison(gat["tasks"], mlp["tasks"]))
        cell["gat_mean"] = cell.pop("mean_a")
        cell["mlp_mean"] = cell.pop("mean_b")
        pibt = _pibt(dist, rate)
        cell["pibt_mean"] = pibt
        cell["gat_over_pibt"] = (
            round(cell["gat_mean"] / pibt, 2) if pibt not in (None, 0) else None
        )
        cell["significant_5pct"] = bool(cell["welch_p"] < 0.05)
        cells.append(cell)
    cells.sort(key=lambda c: (c["distribution"], c["task_rate"]))
    return {"comparison": "GAT_vs_MLP_welch", "cells": cells}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, default=None)
    parser.add_argument("--distribution", default=None)
    parser.add_argument("--task-rate", type=float, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(30)))
    parser.add_argument("--num-agents", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--encoder", choices=["mlp", "gat"], default="gat")
    parser.add_argument("--results-json", type=Path, default=None)
    parser.add_argument(
        "--aggregate",
        type=Path,
        default=None,
        metavar="DIR",
        help="Skip eval; recompute GAT-vs-MLP Welch stats from the per-run "
        "JSONs in DIR and write DIR/stats_summary.json.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.aggregate is not None:
        stats = aggregate(args.aggregate)
        out = args.aggregate / "stats_summary.json"
        out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        sig = sum(1 for c in stats["cells"] if c["significant_5pct"])
        print(
            f"[load_sweep] aggregated {len(stats['cells'])} cells, "
            f"{sig} significant at p<0.05 -> wrote {out}"
        )
        return 0
    if args.actor is None or args.distribution is None or args.task_rate is None:
        build_parser().error(
            "--actor, --distribution, --task-rate are required unless --aggregate"
        )
    s = eval_throughput(
        args.actor, args.distribution, args.seeds, args.task_rate,
        num_agents=args.num_agents, horizon=args.horizon, encoder=args.encoder,
    )
    print(
        f"[load_sweep] {args.encoder} {s['distribution']} lambda={args.task_rate}: "
        f"throughput {s['throughput_mean']} +/- {s['throughput_std']} (n={s['n']})"
    )
    if args.results_json is not None:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(json.dumps(s, indent=2), encoding="utf-8")
        print(f"[load_sweep] wrote {args.results_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
