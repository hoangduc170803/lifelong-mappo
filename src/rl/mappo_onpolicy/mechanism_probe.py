"""Sprint 5 P0 — mechanism probe: MEASURE supply- vs execution-limited (not infer).

The headroom NO-GO rests on a mechanism claim: at the operating load, throughput
is capped by **supply** (hotspot has slack and clears nearly all work) or by
**execution/congestion** (burst), NOT by **assignment** — so no dispatcher can
move it. Previously that claim was *inferred* from throughput behaviour; reviewers
(R1/R3/DA) asked for it to be *measured*. This probe measures it directly.

Per step (AFTER dispatch, so idle agents are genuine residual slack) it records,
for greedy vs Hungarian on hotspot vs burst:
  - mean #idle agents  -> agent utilization = 1 - idle/N
        idle > 0 / util < 1  => spare capacity  (supply/slack-limited)
        idle ~ 0 / util ~ 1  => all agents busy (execution-limited)
  - pool occupancy = (pending + in_flight) / max_pool_size
        low  => queue clears                    (supply-limited)
        ~1.0 => queue pinned at the cap, tasks dropped (execution/cap-saturated)
  - clearance ratio = completed / admitted-arrivals (completed+in_flight+pending)
        ~1.0 => clears almost all admitted work (supply-limited)
        < 1  => leaves a backlog                (execution-limited)

KEY TEST: if these signals are ~IDENTICAL under greedy and Hungarian, the binding
constraint is the SAME regardless of assignment policy -> assignment is not the
bottleneck, corroborating NO-GO from the mechanism (not just the null CI).

NOTE (admitted arrivals): tasks the generator drops at the pool cap are never
created, so `arrivals` counts ADMITTED demand only; if pool occupancy is ~1.0 the
true demand is higher and the system is even more execution-limited than the
clearance ratio shows.

    python -m src.rl.mappo_onpolicy.mechanism_probe \
        --curriculum-dir results/sprint4b/curriculum \
        --distributions hotspot burst_wave --seeds 0..9
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch

_ON_POLICY_ROOT = Path(__file__).resolve().parents[3] / "on-policy"
if str(_ON_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(_ON_POLICY_ROOT))

from src.baselines.hungarian_dispatch import hungarian_dispatch
from src.rl.mappo_onpolicy.env_adapter import WarehouseEnvConfig, WarehouseOnPolicyEnv
from src.rl.mappo_onpolicy.gate_eval import _eval_args, load_actor

DEFAULT_HORIZON = 1024

# None == built-in greedy (env default).
DISPATCHERS: dict[str, Optional[Callable]] = {
    "greedy": None,
    "hungarian": hungarian_dispatch,
}


def _episode_stats(
    idle_series: Sequence[int],
    pending_series: Sequence[int],
    inflight_series: Sequence[int],
    completions: int,
    arrivals: int,
    max_pool: int,
) -> dict[str, float]:
    """Pure aggregation of one episode's per-step series into mechanism signals.

    Kept actor-free so the mechanism logic is unit-testable without a checkpoint.
    """
    horizon = len(idle_series)
    occ = [
        (p + f) / max_pool if max_pool > 0 else 0.0
        for p, f in zip(pending_series, inflight_series)
    ]
    xs = np.arange(horizon)
    # backlog trend: slope of pending over the episode (tasks/step); a binding
    # pool cap flattens this, so it is secondary to pool occupancy.
    slope = float(np.polyfit(xs, pending_series, 1)[0]) if horizon > 1 else 0.0
    return {
        "mean_idle": statistics.mean(idle_series) if idle_series else float("nan"),
        "mean_pending": statistics.mean(pending_series) if pending_series else float("nan"),
        "mean_pool_occ": statistics.mean(occ) if occ else float("nan"),
        "pending_slope": slope,
        "arrivals": float(arrivals),
        "completions": float(completions),
        "clear_ratio": completions / arrivals if arrivals > 0 else float("nan"),
    }


def _actor_path(curriculum_dir: Path, dist: str) -> Path:
    return (
        curriculum_dir / "seed0" / f"n20_{dist}"
        / f"n20_{dist}_seed0" / "models" / "actor.pt"
    )


def mechanism_run(
    actor: Path,
    distribution: str,
    seeds: Sequence[int],
    dispatcher: Optional[Callable] = None,
    task_rate: float = 0.1,
    num_agents: int = 20,
    horizon: int = DEFAULT_HORIZON,
    encoder: str = "gat",
    device: Optional[torch.device] = None,
) -> dict[str, object]:
    """Frozen actor + ``dispatcher``; per-step idle / pending / in_flight, then
    aggregate the supply-vs-execution mechanism signals over ``seeds``."""
    device = device or torch.device("cpu")
    args = _eval_args(num_agents, encoder=encoder)
    cfg = WarehouseEnvConfig(
        num_agents=num_agents, episode_horizon=horizon, task_rate=task_rate,
        task_distribution=distribution, knn_agents=args.knn_agents,
    )
    wrap = WarehouseOnPolicyEnv(cfg, auto_reset=False)
    wrap.env.set_dispatcher(dispatcher)
    actor_net = load_actor(
        actor, args, wrap.observation_space[0], wrap.action_space[0], device
    )
    hidden, rec_n = args.hidden_size, args.recurrent_N
    max_pool = int(wrap.env._task_gen.max_pool_size) if wrap.env._task_gen else 50

    per_seed: list[dict[str, float]] = []
    for seed in seeds:
        obs, _, avail = wrap.reset(seed=seed)
        max_pool = int(wrap.env._task_gen.max_pool_size)
        rnn = np.zeros((num_agents, rec_n, hidden), dtype=np.float32)
        masks = np.ones((num_agents, 1), dtype=np.float32)
        idle_s, pend_s, inflight_s = [], [], []
        completed = 0
        for _ in range(horizon):
            with torch.no_grad():
                actions, _, _ = actor_net(obs, rnn, masks, avail, deterministic=True)
            obs, _, _, dones, info, avail = wrap.step(actions.detach().cpu().numpy())
            completed = int(info["tasks_completed_total"])
            tg = wrap.env._task_gen
            idle_s.append(sum(1 for a in wrap.env._agent_info.values() if a.task is None))
            pend_s.append(len(tg.pending))
            inflight_s.append(len(tg.in_flight))
            if bool(np.asarray(dones).all()):
                break
        tg = wrap.env._task_gen
        arrivals = len(tg.completed) + len(tg.in_flight) + len(tg.pending)
        per_seed.append(
            _episode_stats(idle_s, pend_s, inflight_s, completed, arrivals, max_pool)
        )
    wrap.close()

    keys = list(per_seed[0].keys())
    agg = {k: round(statistics.mean(d[k] for d in per_seed), 3) for k in keys}
    agg["agent_util"] = round(1.0 - agg["mean_idle"] / num_agents, 3)
    agg["max_pool_size"] = max_pool
    agg["num_agents"] = num_agents
    agg["n"] = len(per_seed)
    return agg


def _regime(stats: dict[str, object]) -> str:
    """Descriptive label from the measured signals (not a hypothesis test)."""
    clear = stats["clear_ratio"]
    occ = stats["mean_pool_occ"]
    util = stats["agent_util"]
    if clear >= 0.85 and occ < 0.4 and util < 0.95:
        return "SUPPLY/SLACK-limited (clears work, spare capacity)"
    if clear < 0.75 or occ >= 0.6 or util >= 0.97:
        return "EXECUTION-limited (backlog / agents saturated)"
    return "MIXED"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum-dir", type=Path,
                        default=Path("results") / "sprint4b" / "curriculum")
    parser.add_argument("--distributions", nargs="+", default=["hotspot", "burst_wave"])
    parser.add_argument("--dispatchers", nargs="+", default=["greedy", "hungarian"],
                        choices=list(DISPATCHERS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--task-rate", type=float, default=0.1)
    parser.add_argument("--num-agents", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--encoder", choices=["mlp", "gat"], default="gat")
    parser.add_argument("--results-json", type=Path,
                        default=Path("results") / "sprint5" / "mechanism_probe.json")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out: dict[str, object] = {
        "task_rate": args.task_rate, "n": len(args.seeds), "distributions": {}
    }
    for dist in args.distributions:
        actor = _actor_path(args.curriculum_dir, dist)
        out["distributions"][dist] = {}
        print(f"\n=== {dist} (lambda={args.task_rate}, n={len(args.seeds)}) ===", flush=True)
        for disp in args.dispatchers:
            s = mechanism_run(
                actor, dist, args.seeds, dispatcher=DISPATCHERS[disp],
                task_rate=args.task_rate, num_agents=args.num_agents,
                horizon=args.horizon, encoder=args.encoder,
            )
            s["regime"] = _regime(s)
            out["distributions"][dist][disp] = s
            print(
                f"  {disp:9s}: idle {s['mean_idle']:.1f}/{s['num_agents']} "
                f"(util {s['agent_util']:.2f})  pending {s['mean_pending']:.1f}  "
                f"pool_occ {s['mean_pool_occ']:.2f}  slope {s['pending_slope']:+.3f}  "
                f"clear {s['completions']:.0f}/{s['arrivals']:.0f} "
                f"({s['clear_ratio']:.2f})  -> {s['regime']}",
                flush=True,
            )
        # Same-constraint check. The BINDING CONSTRAINT is defined by the
        # capacity signals (agent utilization + pool occupancy), NOT by
        # clear_ratio/completions — those are the throughput OUTCOME, which can
        # nudge within a fixed regime (e.g. Hungarian's small burst gain that the
        # n=30 confirm showed is not significant). Compare the constraint signals.
        ds = out["distributions"][dist]
        if "greedy" in ds and "hungarian" in ds:
            g, h = ds["greedy"], ds["hungarian"]
            util_d = abs(g["agent_util"] - h["agent_util"])
            occ_d = abs(g["mean_pool_occ"] - h["mean_pool_occ"])
            same = util_d < 0.03 and occ_d < 0.05
            tput_nudge = round(h["completions"] - g["completions"], 1)
            out.setdefault("constraint_comparison", {})[dist] = {
                "constraint_unchanged_by_assignment": bool(same),
                "util_diff": round(util_d, 3),
                "pool_occ_diff": round(occ_d, 3),
                "hungarian_minus_greedy_completions": tput_nudge,
            }
            print(
                f"  -> binding constraint "
                f"{'UNCHANGED by assignment' if same else 'DIFFERS'} "
                f"(util d{util_d:.2f}, pool_occ d{occ_d:.2f}); Hungarian completes "
                f"{tput_nudge:+.0f} within the regime "
                f"(n={len(args.seeds)} -> see n=30 confirm for significance)",
                flush=True,
            )

    if args.results_json is not None:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\n[mechanism_probe] wrote {args.results_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
