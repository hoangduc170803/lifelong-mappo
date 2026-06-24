"""Entry point for the warehouse MAPPO training run (on-policy backend).

Usage example:

    python -m src.rl.mappo_onpolicy.train \
        --num_agents_target 5 --episode_length 128 \
        --num_env_steps 200000 --experiment_name sprint3_smoke

Logs land under ``--results_dir/<experiment_name>_seed<seed>/`` containing
``logs/`` (tensorboardX scalars) and ``models/`` (actor.pt / critic.pt).
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Auto-insert the vendored on-policy package onto sys.path so users do not
# need to set PYTHONPATH manually before launching this entrypoint. The
# package itself lives at the repo root (`on-policy/`); see PATCHES.md.
_ON_POLICY_ROOT = Path(__file__).resolve().parents[3] / "on-policy"
if str(_ON_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(_ON_POLICY_ROOT))

from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy  # noqa: E402
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO  # noqa: E402

from src.rl.mappo_onpolicy.config import get_warehouse_config
from src.rl.mappo_onpolicy.env_adapter import (
    WarehouseEnvConfig,
    WarehouseOnPolicyEnv,
)
from src.rl.mappo_onpolicy.vec_wrapper import DummyVecEnv
from src.rl.mappo_onpolicy.warehouse_runner import WarehouseRunner


def _seed_all(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _make_vec_envs(all_args, num_threads: int) -> DummyVecEnv:
    horizon = (
        all_args.episode_horizon
        if all_args.episode_horizon is not None
        else all_args.episode_length
    )

    def make(rank: int) -> WarehouseOnPolicyEnv:
        cfg = WarehouseEnvConfig(
            map_path=Path(all_args.map_file),
            num_agents=all_args.num_agents_target,
            episode_horizon=horizon,
            task_rate=all_args.task_rate,
            knn_agents=all_args.knn_agents,
            lookahead_action_mask=not all_args.disable_lookahead_action_mask,
            task_distribution=all_args.task_distribution,
            pending_task_features_k=all_args.pending_task_features_k,
            task_budget=all_args.task_budget,
            reward_throughput_bonus_per_task=(
                all_args.reward_throughput_bonus_per_task
            ),
            reward_idle_when_tasks_pending=all_args.reward_idle_when_tasks_pending,
            seed=all_args.seed + 1000 * rank,
        )
        return WarehouseOnPolicyEnv(cfg)

    return DummyVecEnv([make for _ in range(num_threads)])


def main(argv: list[str] | None = None) -> int:
    parser = get_warehouse_config()
    all_args = parser.parse_args(argv)

    if all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy:
        raise ValueError(
            "Sprint 3 expects --use_recurrent_policy False"
            " (vanilla MLP); enable later in Sprint 4 if needed."
        )
    if all_args.use_popart and all_args.use_valuenorm:
        raise ValueError("Set exactly one of --use_popart / --use_valuenorm.")

    _seed_all(all_args.seed, all_args.cuda_deterministic)

    want_cuda = all_args.cuda or getattr(all_args, "use_gpu", False)
    device = torch.device("cuda" if want_cuda and torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(all_args.results_dir) / (
        f"{all_args.experiment_name}_seed{all_args.seed}"
    )
    log_dir = run_dir / "logs"
    save_dir = run_dir / "models"
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[mappo_onpolicy] run dir: {run_dir}")
    print(f"[mappo_onpolicy] device: {device}")
    print(
        f"[mappo_onpolicy] agents={all_args.num_agents_target} "
        f"episode_length={all_args.episode_length} "
        f"num_env_steps={int(all_args.num_env_steps)} "
        f"task_distribution={all_args.task_distribution}"
    )

    envs = _make_vec_envs(all_args, all_args.n_rollout_threads)
    eval_envs = (
        _make_vec_envs(all_args, all_args.n_eval_rollout_threads)
        if all_args.use_eval
        else None
    )

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": all_args.num_agents_target,
        "device": device,
        "run_dir": run_dir,
    }

    runner = WarehouseRunner(config)

    # Curriculum warm-start: load ONLY the actor from a previous stage. The
    # shared GNN actor is N-invariant (knn fixed -> obs_dim fixed), so it
    # transfers across agent-count stages; the centralized critic's cent_obs
    # dimension grows with N, so it is left freshly initialized each stage.
    actor_init = getattr(all_args, "actor_init_from", None)
    if actor_init:
        state = torch.load(str(actor_init), map_location=device)
        runner.policy.actor.load_state_dict(state)
        print(f"[mappo_onpolicy] warm-started actor from {actor_init}")

    try:
        runner.run()
    finally:
        envs.close()
        if eval_envs is not None:
            eval_envs.close()
        if runner.use_wandb:
            try:
                import wandb

                wandb.finish()
            except Exception:  # pragma: no cover - wandb optional
                pass
        else:
            runner.writter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
