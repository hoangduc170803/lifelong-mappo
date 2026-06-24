"""Argparse config for the warehouse MAPPO training entry point.

We extend the upstream `onpolicy.config.get_config` parser with the
warehouse-specific knobs (map path, agents, task rate, ...) and override
defaults that fit a single-process MLP MAPPO smoke training run on CPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from onpolicy.config import get_config

from src.rl.mappo_onpolicy.env_adapter import DEFAULT_MAP_FILE


def get_warehouse_config() -> argparse.ArgumentParser:
    """Return the warehouse-aware argument parser."""
    parser = get_config()
    parser.set_defaults(
        algorithm_name="mappo",
        env_name="warehouse",
        experiment_name="sprint3_smoke",
        seed=0,
        cuda=False,
        cuda_deterministic=True,
        n_rollout_threads=1,
        n_eval_rollout_threads=1,
        num_env_steps=200_000,
        episode_length=128,
        use_recurrent_policy=False,
        use_naive_recurrent_policy=False,
        recurrent_N=1,
        hidden_size=64,
        layer_N=2,
        use_ReLU=False,  # default to Tanh to match the rest of the codebase
        use_popart=False,
        use_valuenorm=True,
        use_feature_normalization=True,
        use_orthogonal=True,
        gain=0.01,
        use_centralized_V=True,
        share_policy=True,
        lr=5e-4,
        critic_lr=5e-4,
        opti_eps=1e-5,
        weight_decay=0.0,
        ppo_epoch=4,
        num_mini_batch=4,
        clip_param=0.2,
        entropy_coef=0.03,
        value_loss_coef=1.0,
        max_grad_norm=10.0,
        gamma=0.99,
        gae_lambda=0.95,
        use_clipped_value_loss=True,
        use_huber_loss=True,
        huber_delta=10.0,
        use_value_active_masks=True,
        use_policy_active_masks=True,
        use_linear_lr_decay=False,
        save_interval=10,
        log_interval=1,
        use_eval=False,
        use_wandb=False,
        user_name="ippo_mapf",
    )

    group = parser.add_argument_group("warehouse")
    group.add_argument(
        "--map_file",
        type=Path,
        default=DEFAULT_MAP_FILE,
        help="Path to the OpenTCS warehouse XML map (defaults to the project root).",
    )
    group.add_argument(
        "--num_agents_target",
        type=int,
        default=5,
        help="Number of AGVs in the warehouse env (default: 5).",
    )
    group.add_argument(
        "--episode_horizon",
        type=int,
        default=None,
        help=(
            "Per-env truncation horizon (defaults to --episode_length so each"
            " rollout is one episode)."
        ),
    )
    group.add_argument(
        "--task_rate",
        type=float,
        default=0.1,
        help="Poisson rate for new pickup-delivery tasks per step (default: 0.1).",
    )
    group.add_argument(
        "--task_distribution",
        choices=[
            "uniform",
            "hotspot",
            "burst_wave",
            "betweenness_bottleneck",
            "bipartite_pickup_dropoff",
        ],
        default="uniform",
        help="Lifelong MAPD task endpoint distribution for Sprint 4a.",
    )
    group.add_argument(
        "--knn_agents",
        type=int,
        default=3,
        help="Number of nearest neighbours encoded in each agent's observation.",
    )
    group.add_argument(
        "--encoder",
        choices=["mlp", "gat"],
        default="mlp",
        help=(
            "Observation encoder for actor/critic. 'mlp' = upstream MLPBase; "
            "'gat' = GATv2 over the kNN agent graph (Sprint 4b, requires "
            "torch_geometric)."
        ),
    )
    group.add_argument(
        "--gat_heads",
        type=int,
        default=4,
        help="Attention heads for the --encoder gat GATv2 layer.",
    )
    group.add_argument(
        "--gnn_node_dim",
        type=int,
        default=32,
        help="Per-node embedding width for the --encoder gat layer.",
    )
    group.add_argument(
        "--pending_task_features_k",
        type=int,
        default=4,
        help="Number of nearest pending tasks appended to centralized critic state.",
    )
    group.add_argument(
        "--use_gpu",
        action="store_true",
        help=(
            "Train on CUDA if available. (Upstream --cuda is store_false with "
            "an overridden default, so it cannot enable the GPU from the CLI; "
            "this flag sets cuda=True post-parse.)"
        ),
    )
    group.add_argument(
        "--actor_init_from",
        type=Path,
        default=None,
        help=(
            "Path to an actor.pt to warm-start ONLY the actor (curriculum "
            "stage transfer). The critic is always freshly initialized "
            "(its cent_obs dim grows with agent count)."
        ),
    )
    group.add_argument(
        "--task_budget",
        type=int,
        default=None,
        help=(
            "Lifelong eval regime: terminate the episode (success) once this "
            "many tasks complete. Default None = training regime (step-horizon "
            "truncation only)."
        ),
    )
    group.add_argument(
        "--reward_throughput_bonus_per_task",
        type=float,
        default=0.0,
        help="Global bonus added to each agent per task completed in the step.",
    )
    group.add_argument(
        "--reward_idle_when_tasks_pending",
        type=float,
        default=0.0,
        help="Per-agent idle penalty applied when pending tasks remain after dispatch.",
    )
    group.add_argument(
        "--disable_lookahead_action_mask",
        action="store_true",
        help=(
            "Disable the conservative one-step collision look-ahead mask. "
            "By default, action masks remove graph-invalid moves plus moves "
            "into occupied/predicted-occupied nodes."
        ),
    )
    group.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results") / "sprint3" / "onpolicy_smoke",
        help="Directory for tensorboard logs and model checkpoints.",
    )
    return parser
