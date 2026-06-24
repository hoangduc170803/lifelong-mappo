"""Sprint 5 Bậc 0 — lifelong dispatcher eval (frozen MAPPO L2 + injected dispatcher).

Runs a trained MAPPO Layer-2 router (movement, FROZEN) with a pluggable Layer-1
dispatcher (greedy / Hungarian / oracle / learned) injected via the env dispatch
hook, and measures throughput + task wait times. Harness for the GATE-2 baseline
comparison and the headroom probe: every dispatcher runs on the SAME frozen
Layer 2 and the SAME task seed, so differences isolate **assignment quality**
(metric re-tightened — continuous throughput / p95 wait + paired CI, NOT Wilson;
see gate2_metric_decision).

The frozen actor is goal-conditioned (its obs encodes dist/next-hint to the
agent's *assigned* task), so swapping the dispatcher only changes which goal each
agent receives; the actor moves toward it regardless of who assigned it.

    python -m src.rl.mappo_onpolicy.dispatcher_eval \
        --actor .../n20_hotspot_seed0/models/actor.pt --distribution hotspot \
        --dispatcher hungarian --task-rate 0.1 --seeds 0..9
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

# Named dispatchers selectable from the CLI. None == built-in greedy.
DISPATCHERS: dict[str, Optional[Callable]] = {
    "greedy": None,
    "hungarian": hungarian_dispatch,
}


def eval_dispatcher(
    actor: Path,
    distribution: str,
    seeds: Sequence[int],
    dispatcher: Optional[Callable] = None,
    task_rate: float = 0.1,
    num_agents: int = 20,
    horizon: int = DEFAULT_HORIZON,
    encoder: str = "gat",
    window: int = 0,
    device: Optional[torch.device] = None,
) -> dict[str, object]:
    """Run the frozen actor + ``dispatcher`` over ``seeds``; per-seed throughput
    and completed-task wait (spawn->delivery latency).

    ``window`` (steps): if >0, also record the per-window completion count
    (tasks cleared in each ``window``-step slice) per seed, so a long-horizon
    run exposes whether throughput is a flat steady state after the initial-pool
    burst (washout) or a declining trajectory (degradation) — the endurance
    probe's per-window curve (verdict.md §10)."""
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
    wrap.env.set_dispatcher(dispatcher)  # None -> greedy
    actor_net = load_actor(
        actor, args, wrap.observation_space[0], wrap.action_space[0], device
    )
    hidden, rec_n = args.hidden_size, args.recurrent_N

    throughput: list[int] = []
    avg_waits: list[float] = []
    p95_waits: list[float] = []
    per_window: list[list[int]] = []  # per-seed list of per-window completion deltas
    for seed in seeds:
        obs, _, avail = wrap.reset(seed=seed)
        rnn = np.zeros((num_agents, rec_n, hidden), dtype=np.float32)
        masks = np.ones((num_agents, 1), dtype=np.float32)
        completed = 0
        win_cum: list[int] = []  # cumulative completions at each full window boundary
        for step_idx in range(horizon):
            with torch.no_grad():
                actions, _, _ = actor_net(obs, rnn, masks, avail, deterministic=True)
            obs, _, _, dones, info, avail = wrap.step(actions.detach().cpu().numpy())
            completed = int(info["tasks_completed_total"])
            if window and (step_idx + 1) % window == 0:
                win_cum.append(completed)
            if bool(np.asarray(dones).all()):
                break
        waits = [
            t.completed_step - t.spawn_step
            for t in wrap.env._task_gen.completed
            if t.completed_step is not None
        ]
        throughput.append(completed)
        if waits:
            avg_waits.append(float(statistics.mean(waits)))
            p95_waits.append(float(np.percentile(waits, 95)))
        if window:
            # cumulative snapshots -> per-window deltas (tasks cleared per slice)
            deltas = [win_cum[0]] if win_cum else []
            deltas += [win_cum[i] - win_cum[i - 1] for i in range(1, len(win_cum))]
            per_window.append(deltas)
    wrap.close()

    return {
        "distribution": distribution,
        "dispatcher": getattr(dispatcher, "__name__", "greedy"),
        "task_rate": task_rate,
        "n": len(throughput),
        "throughput_mean": round(statistics.mean(throughput), 2),
        "throughput_std": round(statistics.pstdev(throughput), 2) if len(throughput) > 1 else 0.0,
        "throughput": throughput,
        "avg_wait_mean": round(statistics.mean(avg_waits), 2) if avg_waits else None,
        "p95_wait_mean": round(statistics.mean(p95_waits), 2) if p95_waits else None,
        "seeds": list(seeds),
        "window": window or None,
        "per_window_throughput": per_window or None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--distribution", required=True)
    parser.add_argument("--dispatcher", choices=list(DISPATCHERS), default="greedy")
    parser.add_argument("--task-rate", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--num-agents", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--encoder", choices=["mlp", "gat"], default="gat")
    parser.add_argument(
        "--window",
        type=int,
        default=0,
        help=(
            "If >0, also log per-window throughput (tasks cleared per WINDOW-step "
            "slice, per seed) — exposes washout-vs-degradation on long-horizon "
            "endurance runs (verdict.md §10). Default 0 = off."
        ),
    )
    parser.add_argument("--results-json", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    s = eval_dispatcher(
        args.actor, args.distribution, args.seeds,
        dispatcher=DISPATCHERS[args.dispatcher], task_rate=args.task_rate,
        num_agents=args.num_agents, horizon=args.horizon, encoder=args.encoder,
        window=args.window,
    )
    print(
        f"[dispatcher_eval] {args.dispatcher} {s['distribution']} "
        f"lambda={args.task_rate}: throughput {s['throughput_mean']} "
        f"+/- {s['throughput_std']} | p95_wait {s['p95_wait_mean']} (n={s['n']})"
    )
    if args.results_json is not None:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(json.dumps(s, indent=2), encoding="utf-8")
        print(f"[dispatcher_eval] wrote {args.results_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
