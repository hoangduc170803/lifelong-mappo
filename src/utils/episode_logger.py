"""Episode-level metrics logger for warehouse MAPD experiments."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Optional

from src.env.compass_mapper import WAIT_SLOT


@dataclass
class EpisodeMetrics:
    """Aggregated metrics for one rollout."""

    steps: int = 0
    last_completion_step: int = 0
    instance_makespan: int = 0
    tasks_completed: int = 0
    tasks_generated: int = 0
    success_rate: float = 0.0
    throughput_tasks_per_step: float = 0.0
    throughput_tasks_per_1000_steps: float = 0.0
    waiting_time_agent_steps: int = 0
    avg_waiting_time_per_agent: float = 0.0
    total_reward: float = 0.0
    avg_reward_per_agent_step: float = 0.0
    validator_interventions: int = 0
    conflicts_vertex: int = 0
    conflicts_edge_swap: int = 0
    conflicts_following: int = 0
    conflicts_total: int = 0
    elapsed_s: float = 0.0
    steps_per_second: float = 0.0
    extra: dict[str, float | int | str] = field(default_factory=dict)


class EpisodeLogger:
    """Accumulate rollout metrics from `WarehouseEnv.step()` outputs."""

    def __init__(self, num_agents: int):
        if num_agents < 1:
            raise ValueError("num_agents must be >= 1")
        self.num_agents = num_agents
        self.reset()

    def reset(self) -> None:
        self._start_time = time.perf_counter()
        self.steps = 0
        self.waiting_time_agent_steps = 0
        self.total_reward = 0.0
        self.validator_interventions = 0
        self.conflicts_vertex = 0
        self.conflicts_edge_swap = 0
        self.conflicts_following = 0
        self.tasks_completed = 0
        self.tasks_pending = 0
        self.tasks_in_flight = 0
        self._last_completion_step = 0
        self._agent_steps = 0

    def record_step(
        self,
        actions: Mapping[str, int],
        rewards: Mapping[str, float],
        infos: Mapping[str, Mapping[str, int]],
        active_agents: Optional[int] = None,
    ) -> None:
        self.steps += 1
        n_agents = active_agents if active_agents is not None else len(actions)
        self._agent_steps += n_agents
        self.waiting_time_agent_steps += sum(
            1 for action in actions.values() if int(action) == WAIT_SLOT
        )
        self.total_reward += sum(float(reward) for reward in rewards.values())

        # WarehouseEnv currently emits the same step-level info dict for every
        # agent, so taking any entry is intentional.
        step_info = next(iter(infos.values()), {})
        self.validator_interventions += int(step_info.get("validator_interventions", 0))
        self.conflicts_vertex += int(step_info.get("conflicts_vertex", 0))
        self.conflicts_edge_swap += int(step_info.get("conflicts_edge_swap", 0))
        self.conflicts_following += int(step_info.get("conflicts_following", 0))

        completed = int(step_info.get("tasks_completed_total", self.tasks_completed))
        if completed > self.tasks_completed:
            self._last_completion_step = self.steps
        self.tasks_completed = completed
        self.tasks_pending = int(step_info.get("tasks_pending", self.tasks_pending))
        self.tasks_in_flight = int(step_info.get("tasks_in_flight", self.tasks_in_flight))

    def finalize(
        self,
        extra: Optional[dict[str, float | int | str]] = None,
        instance_makespan: Optional[int] = None,
    ) -> EpisodeMetrics:
        """Return aggregated rollout metrics.

        For one-shot MAPF/CBS benchmarks, pass
        `instance_makespan=plan_result.makespan`. Lifelong MAPD rollouts use
        `last_completion_step`; leaving `instance_makespan` unset records 0.
        """
        elapsed = time.perf_counter() - self._start_time
        tasks_generated = self.tasks_completed + self.tasks_pending + self.tasks_in_flight
        conflicts_total = (
            self.conflicts_vertex + self.conflicts_edge_swap + self.conflicts_following
        )
        return EpisodeMetrics(
            steps=self.steps,
            last_completion_step=self._last_completion_step,
            instance_makespan=int(instance_makespan or 0),
            tasks_completed=self.tasks_completed,
            tasks_generated=tasks_generated,
            success_rate=(
                self.tasks_completed / tasks_generated if tasks_generated > 0 else 0.0
            ),
            throughput_tasks_per_step=(
                self.tasks_completed / self.steps if self.steps > 0 else 0.0
            ),
            throughput_tasks_per_1000_steps=(
                1000.0 * self.tasks_completed / self.steps if self.steps > 0 else 0.0
            ),
            waiting_time_agent_steps=self.waiting_time_agent_steps,
            avg_waiting_time_per_agent=(
                self.waiting_time_agent_steps / self.num_agents
                if self.num_agents > 0
                else 0.0
            ),
            total_reward=self.total_reward,
            avg_reward_per_agent_step=(
                self.total_reward / self._agent_steps if self._agent_steps > 0 else 0.0
            ),
            validator_interventions=self.validator_interventions,
            conflicts_vertex=self.conflicts_vertex,
            conflicts_edge_swap=self.conflicts_edge_swap,
            conflicts_following=self.conflicts_following,
            conflicts_total=conflicts_total,
            elapsed_s=elapsed,
            steps_per_second=self.steps / elapsed if elapsed > 0 else 0.0,
            extra=extra or {},
        )


def write_metrics_json(metrics: EpisodeMetrics, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
    return path


def append_metrics_csv(metrics: EpisodeMetrics, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = asdict(metrics)
    extra = row.pop("extra")
    for key, value in extra.items():
        row[f"extra_{key}"] = value

    fieldnames = list(row.keys())
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return path
