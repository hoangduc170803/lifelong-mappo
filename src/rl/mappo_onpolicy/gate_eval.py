"""Sprint 4b — GATE 1 lifelong success-regime evaluation for trained policies.

Loads a trained GAT actor and runs it greedily (deterministic) in the lifelong
warehouse env at N=20 under the SUCCESS regime pinned in 4b-P2: the episode
terminates (success) once `task_budget` tasks complete, else fails at the step
cap. Per-distribution budgets and the GATE-1 thresholds (PIBT Wilson-95 upper,
pinned BEFORE any MAPPO eval — anti-p-hacking) are baked in as constants.

A policy PASSES condition 3 if its success-rate Wilson-95 lower bound exceeds
the distribution's threshold on >= 2 of the 4 hard distributions.

Example:
    python -m src.rl.mappo_onpolicy.gate_eval \
        --actor results/sprint4b/curriculum/seed0/n20_hotspot/n20_hotspot_seed0/models/actor.pt \
        --distribution hotspot --seeds 0..29
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch

_ON_POLICY_ROOT = Path(__file__).resolve().parents[3] / "on-policy"
if str(_ON_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(_ON_POLICY_ROOT))

from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor  # noqa: E402

from src.baselines.stats_utils import wilson_ci
from src.rl.mappo_onpolicy.config import get_warehouse_config
from src.rl.mappo_onpolicy.env_adapter import (
    WarehouseEnvConfig,
    WarehouseOnPolicyEnv,
)

# 4b-P2 pinned success regime (results/sprint4b/p2_baseline_matrix.md).
# budget = PIBT pre-freeze capacity; threshold = PIBT success Wilson-95 upper.
PINNED = {
    "hotspot": {"budget": 11, "threshold": 0.726},
    "burst_wave": {"budget": 5, "threshold": 0.608},
    "betweenness_bottleneck": {"budget": 11, "threshold": 0.639},
    "bipartite_pickup_dropoff": {"budget": 8, "threshold": 0.577},
}
DEFAULT_STEP_CAP = 4096


def _eval_args(num_agents: int, encoder: str = "gat") -> argparse.Namespace:
    """A fully-populated args namespace matching the curriculum training run."""
    parser = get_warehouse_config()
    # use_recurrent_policy / use_naive_recurrent_policy default to False via the
    # warehouse config set_defaults (they are store_true/false flags that take
    # no value), so they are intentionally omitted here.
    args = parser.parse_args(
        [
            "--encoder", encoder,
            "--num_agents_target", str(num_agents),
            "--hidden_size", "64",
            "--knn_agents", "3",
        ]
    )
    args.num_agents = num_agents
    return args


def load_actor(
    checkpoint: Path,
    args: argparse.Namespace,
    obs_space,
    act_space,
    device: torch.device,
) -> R_Actor:
    actor = R_Actor(args, obs_space, act_space, device)
    actor.load_state_dict(torch.load(str(checkpoint), map_location=device))
    actor.eval()
    return actor


def eval_success(
    checkpoint: Path,
    distribution: str,
    seeds: Sequence[int],
    num_agents: int = 20,
    task_budget: Optional[int] = None,
    step_cap: int = DEFAULT_STEP_CAP,
    task_rate: float = 0.1,
    device: Optional[torch.device] = None,
    encoder: str = "gat",
    dispatcher: Optional[Callable] = None,
    threshold: Optional[float] = None,
) -> dict[str, object]:
    if task_budget is None:
        if distribution not in PINNED:
            raise ValueError(
                f"--task-budget is required for non-pinned distribution "
                f"'{distribution}' (no PINNED budget to fall back on)"
            )
        task_budget = PINNED[distribution]["budget"]
    device = device or torch.device("cpu")
    args = _eval_args(num_agents, encoder=encoder)

    cfg = WarehouseEnvConfig(
        num_agents=num_agents,
        episode_horizon=step_cap,
        task_rate=task_rate,
        task_distribution=distribution,
        task_budget=task_budget,
        knn_agents=args.knn_agents,
    )
    wrap = WarehouseOnPolicyEnv(cfg, auto_reset=False)
    wrap.env.set_dispatcher(dispatcher)  # None -> env greedy (GATE-1 default)
    obs_space = wrap.observation_space[0]
    act_space = wrap.action_space[0]
    actor = load_actor(checkpoint, args, obs_space, act_space, device)

    hidden = args.hidden_size
    rec_n = args.recurrent_N
    rows: list[dict[str, object]] = []
    for seed in seeds:
        obs, _, avail = wrap.reset(seed=seed)
        rnn = np.zeros((num_agents, rec_n, hidden), dtype=np.float32)
        masks = np.ones((num_agents, 1), dtype=np.float32)
        success = False
        steps = 0
        for _ in range(step_cap):
            with torch.no_grad():
                actions, _, _ = actor(obs, rnn, masks, avail, deterministic=True)
            actions_np = actions.detach().cpu().numpy()
            obs, _, _, dones, info, avail = wrap.step(actions_np)
            steps += 1
            if bool(np.asarray(dones).all()):
                success = bool(info.get("task_budget_reached", False))
                break
        rows.append({"seed": int(seed), "success": success, "steps": steps})
    wrap.close()

    n = len(rows)
    succ = sum(int(r["success"]) for r in rows)
    lo, hi = wilson_ci(succ, n)
    if threshold is None:
        threshold = PINNED.get(distribution, {}).get("threshold")
    steps_succ = [int(r["steps"]) for r in rows if r["success"]]
    return {
        "distribution": distribution,
        "checkpoint": str(checkpoint),
        "dispatcher": getattr(dispatcher, "__name__", "greedy"),
        "task_budget": task_budget,
        "step_cap": step_cap,
        "n": n,
        "success_count": succ,
        "success_rate": round(succ / n, 3),
        "success_wilson95": [round(lo, 3), round(hi, 3)],
        "gate_threshold": threshold,
        "passes": (bool(lo > threshold) if threshold is not None else None),
        "steps_to_budget_mean": (
            round(float(np.mean(steps_succ)), 1) if steps_succ else None
        ),
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument(
        "--distribution", required=True,
        help="One of the 4 PINNED dists, or a held-out dist (e.g. corner_exchange) "
        "with an explicit --task-budget. Validated by the env.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(30)))
    parser.add_argument("--num-agents", type=int, default=20)
    parser.add_argument("--task-budget", type=int, default=None)
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Separation bar = Wilson-95 upper of the strongest classical at this "
        "cell. Defaults to the PINNED GATE-1 threshold for the 4 canonical dists; "
        "for a held-out dist, pass it explicitly (else 'passes' is reported null).",
    )
    parser.add_argument("--step-cap", type=int, default=DEFAULT_STEP_CAP)
    parser.add_argument("--encoder", choices=["mlp", "gat"], default="gat")
    parser.add_argument(
        "--dispatcher", choices=["greedy", "hungarian"], default="greedy",
        help="Layer-1 dispatcher (Sprint 5: NO-GO on learned -> Hungarian is the "
        "final system's classical dispatcher). Default greedy = GATE-1 behaviour.",
    )
    parser.add_argument("--results-json", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    dispatcher = None
    if args.dispatcher == "hungarian":
        from src.baselines.hungarian_dispatch import hungarian_dispatch
        dispatcher = hungarian_dispatch
    summary = eval_success(
        args.actor,
        args.distribution,
        args.seeds,
        num_agents=args.num_agents,
        task_budget=args.task_budget,
        step_cap=args.step_cap,
        encoder=args.encoder,
        dispatcher=dispatcher,
        threshold=args.threshold,
    )
    lo, hi = summary["success_wilson95"]
    verdict = {True: "PASS", False: "FAIL", None: "N/A"}[summary["passes"]]
    print(
        f"[gate_eval] {summary['distribution']}: "
        f"success {summary['success_count']}/{summary['n']} "
        f"(rate {summary['success_rate']}, Wilson95 [{lo}, {hi}]) "
        f"vs threshold {summary['gate_threshold']} -> {verdict}"
    )
    if args.results_json is not None:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[gate_eval] wrote {args.results_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
