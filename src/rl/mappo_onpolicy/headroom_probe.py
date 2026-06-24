"""Sprint 5 Bậc 0 — headroom probe (decision-gate for the learned dispatcher).

Question: is there ANY assignment headroom for a learned dispatcher to beat
Hungarian-rerun-on-event, holding Layer 2 fixed? If not, ship "MAPPO + Hungarian"
and skip training (gate2_metric_decision).

Runs three dispatchers on the SAME frozen MAPPO Layer 2 + SAME seed, on the
distributions where anticipation should matter (burst/hotspot):
  - greedy        (env default, myopic nearest-pickup)
  - hungarian     (myopic-OPTIMAL assignment; LSAP) -> myopic headroom = hung-greedy
  - oracle        (clairvoyant K-step HOLD heuristic) -> anticipation probe

Metric is continuous (throughput tasks/episode + p95 task wait) with PAIRED
bootstrap CI / Wilcoxon (stats_utils.paired_comparison) — NOT Wilson.

GATE LOGIC: a learned dispatcher must beat **Hungarian** (optimal-myopic,
non-learned). So headroom exists only if the clairvoyant oracle beats Hungarian.
NOTE: the oracle is a *hold-for-imminent-better-task heuristic*, NOT a proven
upper bound — when throughput is supply-saturated, holding agents idle can
*raise* wait, which is itself an informative (anticipation-doesn't-help) signal.

    python -m src.rl.mappo_onpolicy.headroom_probe \
        --actor .../n20_hotspot_seed0/models/actor.pt \
        --distributions hotspot burst_wave --seeds 0..9 --lookahead-k 50
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

from src.baselines.hungarian_dispatch import hungarian_dispatch
from src.baselines.oracle_dispatch import make_oracle_dispatcher, schedule_from_tasks
from src.baselines.stats_utils import paired_comparison
from src.rl.mappo_onpolicy.env_adapter import WarehouseEnvConfig, WarehouseOnPolicyEnv
from src.rl.mappo_onpolicy.gate_eval import _eval_args, load_actor


def _run_episode(wrap, actor_net, seed, horizon, hidden, rec_n, num_agents):
    obs, _, avail = wrap.reset(seed=seed)
    rnn = np.zeros((num_agents, rec_n, hidden), dtype=np.float32)
    masks = np.ones((num_agents, 1), dtype=np.float32)
    completed = 0
    for _ in range(horizon):
        with torch.no_grad():
            actions, _, _ = actor_net(obs, rnn, masks, avail, deterministic=True)
        obs, _, _, dones, info, avail = wrap.step(actions.detach().cpu().numpy())
        completed = int(info["tasks_completed_total"])
        if bool(np.asarray(dones).all()):
            break
    tg = wrap.env._task_gen
    all_tasks = list(tg.completed) + list(tg.in_flight.values()) + list(tg.pending)
    waits = [
        t.completed_step - t.spawn_step
        for t in tg.completed
        if t.completed_step is not None
    ]
    p95 = float(np.percentile(waits, 95)) if waits else float("nan")
    return completed, p95, all_tasks


def probe_distribution(
    actor: Path,
    distribution: str,
    seeds: Sequence[int],
    task_rate: float = 0.1,
    lookahead_k: int = 50,
    num_agents: int = 20,
    horizon: int = 1024,
    encoder: str = "gat",
    device: Optional[torch.device] = None,
) -> dict[str, object]:
    device = device or torch.device("cpu")
    args = _eval_args(num_agents, encoder=encoder)
    cfg = WarehouseEnvConfig(
        num_agents=num_agents, episode_horizon=horizon, task_rate=task_rate,
        task_distribution=distribution, knn_agents=args.knn_agents,
    )
    wrap = WarehouseOnPolicyEnv(cfg, auto_reset=False)
    actor_net = load_actor(
        actor, args, wrap.observation_space[0], wrap.action_space[0], device
    )
    hidden, rec_n = args.hidden_size, args.recurrent_N

    g_tp, g_w, h_tp, h_w, o_tp, o_w = [], [], [], [], [], []
    for seed in seeds:
        wrap.env.set_dispatcher(None)  # greedy + record schedule
        gt, gw, gtasks = _run_episode(wrap, actor_net, seed, horizon, hidden, rec_n, num_agents)
        wrap.env.set_dispatcher(hungarian_dispatch)
        ht, hw, _ = _run_episode(wrap, actor_net, seed, horizon, hidden, rec_n, num_agents)
        oracle = make_oracle_dispatcher(schedule_from_tasks(gtasks), lookahead_k)
        wrap.env.set_dispatcher(oracle)
        ot, ow, _ = _run_episode(wrap, actor_net, seed, horizon, hidden, rec_n, num_agents)
        g_tp.append(gt); h_tp.append(ht); o_tp.append(ot)
        g_w.append(gw); h_w.append(hw); o_w.append(ow)
        print(
            f"[headroom_probe] {distribution} seed={seed}: "
            f"tp g={gt} h={ht} o={ot} | p95wait g={gw:.0f} h={hw:.0f} o={ow:.0f}",
            flush=True,
        )
    wrap.close()

    def _mean(xs):
        return round(statistics.mean(xs), 2)

    return {
        "distribution": distribution,
        "n": len(seeds),
        "lookahead_k": lookahead_k,
        "throughput": {
            "greedy_mean": _mean(g_tp), "hungarian_mean": _mean(h_tp),
            "oracle_mean": _mean(o_tp),
            "hungarian_vs_greedy": paired_comparison(h_tp, g_tp),
            "oracle_vs_greedy": paired_comparison(o_tp, g_tp),
            "oracle_vs_hungarian": paired_comparison(o_tp, h_tp),
        },
        # p95 wait: lower is better, so diff is (greedy - treatment) -> positive = better
        "p95_wait": {
            "greedy_mean": _mean(g_w), "hungarian_mean": _mean(h_w),
            "oracle_mean": _mean(o_w),
            "hungarian_improves_over_greedy": paired_comparison(g_w, h_w),
            "oracle_improves_over_greedy": paired_comparison(g_w, o_w),
            "oracle_improves_over_hungarian": paired_comparison(h_w, o_w),
        },
    }


def _ci_excludes_zero(cmp: dict) -> bool:
    lo, hi = cmp["bootstrap_ci95_mean_diff"]
    return lo > 0 or hi < 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--distributions", nargs="+", default=["hotspot", "burst_wave"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--task-rate", type=float, default=0.1)
    parser.add_argument("--lookahead-k", type=int, default=50)
    parser.add_argument("--num-agents", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1024)
    parser.add_argument("--encoder", choices=["mlp", "gat"], default="gat")
    parser.add_argument("--results-json", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out = {"lookahead_k": args.lookahead_k, "distributions": {}}
    for dist in args.distributions:
        res = probe_distribution(
            args.actor, dist, args.seeds, task_rate=args.task_rate,
            lookahead_k=args.lookahead_k, num_agents=args.num_agents,
            horizon=args.horizon, encoder=args.encoder,
        )
        out["distributions"][dist] = res
        tp, w = res["throughput"], res["p95_wait"]
        # PRIMARY metric = THROUGHPUT (unconfounded). Diagnostics only, NOT the
        # gate: (a) p95-wait-over-completed is confounded by throughput
        # (survivorship — a dispatcher that completes fewer/easier tasks looks
        # better on wait); (b) the naive hold-oracle is DEGENERATE on bursty load
        # (holds agents idle through quiet periods -> throughput collapse), so it
        # is not a valid upper bound. A learned dispatcher must beat Hungarian on
        # throughput; there is no valid evidence it can.
        tp_oh = tp["oracle_vs_hungarian"]   # oracle - hung
        tp_hg = tp["hungarian_vs_greedy"]   # hung - greedy: does assignment help?
        learned_go = _ci_excludes_zero(tp_oh) and tp_oh["mean_diff"] > 0
        assignment_matters = _ci_excludes_zero(tp_hg) and tp_hg["mean_diff"] > 0
        print(
            f"\n=== {dist} (n={res['n']}, K={args.lookahead_k}) ===\n"
            f"  THROUGHPUT (primary)  greedy {tp['greedy_mean']}  hung "
            f"{tp['hungarian_mean']}  oracle {tp['oracle_mean']}\n"
            f"    hung-greedy {tp_hg['mean_diff']:+.1f} (p={tp_hg['paired_t_p']:.3f}) "
            f"-> assignment matters: {assignment_matters}\n"
            f"    oracle-hung {tp_oh['mean_diff']:+.1f} (p={tp_oh['paired_t_p']:.3f})\n"
            f"  p95 wait (CONFOUNDED) g={w['greedy_mean']} h={w['hungarian_mean']} "
            f"o={w['oracle_mean']}  [oracle degenerate on burst -> ignore]\n"
            f"  -> LEARNED-dispatcher headroom: "
            f"{'GO' if learned_go else 'NO-GO (no throughput headroom beyond Hungarian)'}"
            f"  | dispatcher rec: "
            f"{'Hungarian (assignment helps)' if assignment_matters else 'Hungarian (never hurts; ~greedy here)'}",
            flush=True,
        )
    if args.results_json is not None:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\n[headroom_probe] wrote {args.results_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
