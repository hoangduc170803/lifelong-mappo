"""on-policy `Runner` specialization for the warehouse MAPD env.

Modelled after `onpolicy/runner/shared/smac_runner.py`. The warehouse env is
*lifelong* (no episode termination from the env until horizon truncation),
so we treat each rollout of length ``episode_length`` as one episode and
let the env auto-reset on truncation (handled inside `WarehouseOnPolicyEnv`).

The runner pulls:
    obs, share_obs, rewards, dones, infos, available_actions
from the vec env every step and inserts into the shared buffer with active
masks broadcast from the per-agent ``dones`` array.

Logged metrics every ``log_interval`` episodes include the standard PPO
losses plus warehouse-specific aggregates collected from `info`:
``tasks_completed_per_step``, ``validator_intervention_rate``, and per-step
conflict counts (vertex / edge_swap / following), plus
``lookahead_forced_wait_rate`` to catch over-conservative masks at high AGV
density.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from onpolicy.runner.shared.base_runner import Runner

try:  # wandb is optional in the vendored copy
    import wandb
except ImportError:  # pragma: no cover - parity with patched base_runner
    wandb = None


def _t2n(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


class WarehouseRunner(Runner):
    """Runner for the PettingZoo warehouse MAPD env."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        # Cumulative per-rollout warehouse metrics: reset every log_interval.
        self._latest_tasks_completed: int = 0
        self._step_validator_interventions = 0
        self._step_conflicts_vertex = 0
        self._step_conflicts_edge_swap = 0
        self._step_conflicts_following = 0
        self._step_lookahead_forced_wait_rate = 0.0
        self._steps_logged = 0

    # ------------------------------------------------------------ rollout
    def run(self) -> None:
        self.warmup()

        start = time.time()
        episodes = (
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        )

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            episode_info_snapshot: dict[str, Any] = {}
            for step in range(self.episode_length):
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                ) = self.collect(step)

                (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                ) = self.envs.step(actions)
                episode_info_snapshot = infos[-1] if infos else {}

                self._accumulate_step_stats(infos)

                data = (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )
                self.insert(data)

            self.compute()
            train_infos = self.train()

            total_num_steps = (
                (episode + 1) * self.episode_length * self.n_rollout_threads
            )

            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            if episode % self.log_interval == 0:
                end = time.time()
                fps = int(total_num_steps / (end - start)) if end > start else 0
                print(
                    f"\n Env {self.env_name} Algo {self.algorithm_name} "
                    f"Exp {self.experiment_name} updates {episode}/{episodes}, "
                    f"total num timesteps {total_num_steps}/{int(self.num_env_steps)}, "
                    f"FPS {fps}.\n"
                )

                self._log_warehouse_metrics(
                    train_infos,
                    episode_info_snapshot,
                    total_num_steps,
                )
                self._reset_step_stats()

    # -------------------------------------------------------------- helpers
    def warmup(self) -> None:
        obs, share_obs, available_actions = self.envs.reset()
        if not self.use_centralized_V:
            share_obs = obs
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step: int):
        self.trainer.prep_rollout()
        (
            value,
            action,
            action_log_prob,
            rnn_state,
            rnn_state_critic,
        ) = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
            np.concatenate(self.buffer.available_actions[step]),
        )
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_state_critic), self.n_rollout_threads)
        )
        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data: tuple) -> None:
        (
            obs,
            share_obs,
            rewards,
            dones,
            infos,
            available_actions,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(
            (
                (dones_env == True).sum(),
                self.num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        rnn_states_critic[dones_env == True] = np.zeros(
            (
                (dones_env == True).sum(),
                self.num_agents,
                *self.buffer.rnn_states_critic.shape[3:],
            ),
            dtype=np.float32,
        )

        masks = np.ones(
            (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
        )
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        active_masks = np.ones(
            (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
        )
        active_masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1), dtype=np.float32
        )
        active_masks[dones_env == True] = np.ones(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        bad_masks = np.array(
            [
                [
                    [0.0 if info.get("bad_transition", False) else 1.0]
                    for _ in range(self.num_agents)
                ]
                for info in infos
            ],
            dtype=np.float32,
        )

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
            bad_masks,
            active_masks,
            available_actions,
        )

    # ---------------------------------------------------------- logging
    def _accumulate_step_stats(self, infos: list[dict[str, Any]]) -> None:
        for info in infos:
            self._latest_tasks_completed = int(
                info.get("tasks_completed_total", self._latest_tasks_completed)
            )
            self._step_validator_interventions += int(
                info.get("validator_interventions", 0)
            )
            self._step_conflicts_vertex += int(info.get("conflicts_vertex", 0))
            self._step_conflicts_edge_swap += int(info.get("conflicts_edge_swap", 0))
            self._step_conflicts_following += int(info.get("conflicts_following", 0))
            self._step_lookahead_forced_wait_rate += float(
                info.get("lookahead_forced_wait_rate", 0.0)
            )
            self._steps_logged += 1

    def _reset_step_stats(self) -> None:
        self._step_validator_interventions = 0
        self._step_conflicts_vertex = 0
        self._step_conflicts_edge_swap = 0
        self._step_conflicts_following = 0
        self._step_lookahead_forced_wait_rate = 0.0
        self._steps_logged = 0

    def _log_warehouse_metrics(
        self,
        train_infos: dict[str, Any],
        latest_info: dict[str, Any],
        total_num_steps: int,
    ) -> None:
        steps = max(self._steps_logged, 1)
        train_infos["average_step_rewards"] = float(np.mean(self.buffer.rewards))
        train_infos["warehouse/tasks_completed_total"] = float(
            self._latest_tasks_completed
        )
        train_infos["warehouse/validator_intervention_rate"] = (
            self._step_validator_interventions / steps
        )
        train_infos["warehouse/conflicts_vertex_per_step"] = (
            self._step_conflicts_vertex / steps
        )
        train_infos["warehouse/conflicts_edge_swap_per_step"] = (
            self._step_conflicts_edge_swap / steps
        )
        train_infos["warehouse/conflicts_following_per_step"] = (
            self._step_conflicts_following / steps
        )
        train_infos["warehouse/lookahead_forced_wait_rate"] = (
            self._step_lookahead_forced_wait_rate / steps
        )
        train_infos["warehouse/tasks_pending"] = float(
            latest_info.get("tasks_pending", 0)
        )
        train_infos["warehouse/tasks_in_flight"] = float(
            latest_info.get("tasks_in_flight", 0)
        )
        self.log_train(train_infos, total_num_steps)

    def log_train(self, train_infos: dict[str, Any], total_num_steps: int) -> None:
        for key, value in train_infos.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            scalar = float(value)
            if self.use_wandb and wandb is not None:
                wandb.log({key: scalar}, step=total_num_steps)
            else:
                self.writter.add_scalar(key, scalar, total_num_steps)
