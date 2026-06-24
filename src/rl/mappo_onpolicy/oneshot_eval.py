"""Sprint 4b — GATE 1 condition 2: trained MAPPO on ONE-SHOT MAPF vs PIBT/LaCAM.

Evaluates the trained GAT policy on the SAME one-shot instances as P3
(`sprint35_gate.sample_warehouse_stress_instance`), so its success/makespan are
directly comparable to the PIBT/LaCAM bar in
`results/sprint4b/p3_classical_stress.md`.

One-shot is injected into the lifelong env: each agent is given a single
already-picked-up task whose dropoff is the instance goal (so `current_goal`
== goal throughout), the pending pool is cleared and task_rate is 0 (no new
arrivals). An agent is "delivered" the first tick it stands on its goal; once
delivered it is forced to WAIT (parked), exactly as a one-shot solver leaves a
finished agent. Success = all agents delivered within the step cap; makespan =
the tick the last agent arrives.

Example:
    python -m src.rl.mappo_onpolicy.oneshot_eval \
        --actor results/sprint4b/curriculum/seed0/n20_hotspot/.../actor.pt \
        --distribution hotspot --seeds 0..29
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

_ON_POLICY_ROOT = Path(__file__).resolve().parents[3] / "on-policy"
if str(_ON_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(_ON_POLICY_ROOT))

from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor  # noqa: E402

from src.baselines.sprint35_gate import sample_warehouse_stress_instance
from src.baselines.stats_utils import wilson_ci
from src.env.compass_mapper import WAIT_SLOT
from src.env.warehouse_env import AgentState
from src.rl.mappo_onpolicy.env_adapter import WarehouseEnvConfig, build_warehouse_env
from src.rl.mappo_onpolicy.gate_eval import _eval_args, load_actor

DEFAULT_STEP_CAP = 1024
# P3 one-shot classical bar (results/sprint4b/p3_classical_stress.md).
P3_BAR = {
    "hotspot": {"pibt": 1.00, "lacam": 1.00},
    "burst_wave": {"pibt": 1.00, "lacam": 1.00},
    "betweenness_bottleneck": {"pibt": 0.00, "lacam": 0.00},
    "bipartite_pickup_dropoff": {"pibt": 1.00, "lacam": 1.00},
}


def _inject_oneshot(env, starts: dict, goals: dict) -> None:
    """Give every agent a single already-picked-up task dropoff=goal."""
    # reset() pre-seeds the pool and greedy-dispatches, leaving orphan tasks in
    # BOTH pending and in_flight. Clear both so the injected state holds exactly
    # one in_flight task per agent — otherwise num_in_flight() (and thus the
    # critic state + the tasks_in_flight diagnostic) is contaminated by the
    # reset-time orphans whose agents we are about to overwrite.
    env._task_gen.pending.clear()
    env._task_gen.in_flight.clear()
    for a in env.agents:
        info = env._agent_info[a]
        info.pos = info.prev_pos = starts[a]
        task = env._task_gen.create_task(step_idx=0)
        task.pickup, task.dropoff = starts[a], goals[a]
        task.picked_up = True
        task.assigned_to = a
        # Register in_flight so the env's dropoff-completion (pop from
        # in_flight) succeeds when the agent arrives at its goal.
        env._task_gen.in_flight[task.id] = task
        info.task = task
        info.state = AgentState.TO_DROPOFF


def _stack(env):
    obs_batch = env._build_obs_batch()
    order = list(env.agents)
    obs = np.stack([obs_batch[a]["observation"].astype(np.float32) for a in order])
    avail = np.stack([obs_batch[a]["action_mask"].astype(np.float32) for a in order])
    return order, obs, avail


def run_oneshot_instance(env, actor, starts, goals, step_cap, hidden, rec_n):
    env.reset(seed=0)  # placement reset; positions overwritten below
    _inject_oneshot(env, starts, goals)
    agents = list(env.agents)
    delivered = {a: False for a in agents}
    makespan = 0
    for t in range(step_cap):
        for a in agents:
            if env._agent_info[a].pos == goals[a]:
                delivered[a] = True
        if all(delivered.values()):
            makespan = t
            break
        order, obs, avail = _stack(env)
        rnn = np.zeros((len(order), rec_n, hidden), dtype=np.float32)
        masks = np.ones((len(order), 1), dtype=np.float32)
        with torch.no_grad():
            actions, _, _ = actor(obs, rnn, masks, avail, deterministic=True)
        acts = actions.detach().cpu().numpy().reshape(-1).astype(int)
        action_dict = {}
        for i, a in enumerate(order):
            # Park delivered agents (a one-shot solver leaves them at goal).
            action_dict[a] = WAIT_SLOT if delivered[a] else int(acts[i])
        env.step(action_dict)
        makespan = t + 1
    success = all(env._agent_info[a].pos == goals[a] for a in agents)
    return {"success": bool(success), "makespan": makespan if success else 0}


def eval_oneshot(
    checkpoint: Path,
    distribution: str,
    seeds: Sequence[int],
    num_agents: int = 20,
    step_cap: int = DEFAULT_STEP_CAP,
    device: Optional[torch.device] = None,
) -> dict[str, object]:
    device = device or torch.device("cpu")
    args = _eval_args(num_agents)
    cfg = WarehouseEnvConfig(
        num_agents=num_agents,
        episode_horizon=step_cap,
        task_rate=0.0,
        task_distribution="uniform",  # instances injected, not env-sampled
        knn_agents=args.knn_agents,
    )
    env = build_warehouse_env(cfg)
    obs_space = env.observation_space(env.possible_agents[0])["observation"]
    # Build a 1-D Box matching the actor's obs slice.
    from gymnasium import spaces

    obs_box = spaces.Box(
        low=-np.inf, high=np.inf, shape=(obs_space.shape[0],), dtype=np.float32
    )
    actor = load_actor(checkpoint, args, obs_box, env.action_space(env.possible_agents[0]), device)

    rows: list[dict[str, object]] = []
    for seed in seeds:
        starts, goals = sample_warehouse_stress_instance(
            env.G, num_agents, seed, distribution
        )
        res = run_oneshot_instance(
            env, actor, starts, goals, step_cap, args.hidden_size, args.recurrent_N
        )
        res["seed"] = int(seed)
        rows.append(res)

    n = len(rows)
    succ = sum(int(r["success"]) for r in rows)
    lo, hi = wilson_ci(succ, n)
    mk = [int(r["makespan"]) for r in rows if r["success"]]
    return {
        "distribution": distribution,
        "checkpoint": str(checkpoint),
        "regime": "one_shot",
        "n": n,
        "success_count": succ,
        "success_rate": round(succ / n, 3),
        "success_wilson95": [round(lo, 3), round(hi, 3)],
        "makespan_mean": round(float(np.mean(mk)), 1) if mk else None,
        "p3_pibt_success": P3_BAR[distribution]["pibt"],
        "p3_lacam_success": P3_BAR[distribution]["lacam"],
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--distribution", choices=list(P3_BAR), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(30)))
    parser.add_argument("--num-agents", type=int, default=20)
    parser.add_argument("--step-cap", type=int, default=DEFAULT_STEP_CAP)
    parser.add_argument("--results-json", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = eval_oneshot(
        args.actor,
        args.distribution,
        args.seeds,
        num_agents=args.num_agents,
        step_cap=args.step_cap,
    )
    lo, hi = summary["success_wilson95"]
    print(
        f"[oneshot_eval] {summary['distribution']}: MAPPO one-shot "
        f"success {summary['success_count']}/{summary['n']} "
        f"(rate {summary['success_rate']}, Wilson95 [{lo}, {hi}]) "
        f"makespan~{summary['makespan_mean']} | P3 bar PIBT "
        f"{summary['p3_pibt_success']} LaCAM {summary['p3_lacam_success']}"
    )
    if args.results_json is not None:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[oneshot_eval] wrote {args.results_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
