"""Classical assignment baselines for MAPD and one-shot MAPF.

The MAPD variants assign pending pickup/dropoff tasks to agents. The one-shot
MAPF variants assign a set of goal nodes to agents before a path planner runs.
All functions are pure: they do not mutate tasks or env state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.env.task_generator import Task


class DistanceRouter(Protocol):
    """Minimal router interface used by assignment baselines."""

    def distance(self, source: str, target: str) -> float:
        ...


@dataclass(frozen=True)
class TaskAssignment:
    """A task selected for an agent, with distance to pickup."""

    agent: str
    task: Task
    cost_to_pickup: float


@dataclass(frozen=True)
class GoalAssignment:
    """A one-shot MAPF goal selected for an agent."""

    agent: str
    goal: str
    cost: float


def fifo_nearest_task_assignment(
    agent_positions: Mapping[str, str],
    tasks: Sequence[Task],
    router: DistanceRouter,
    max_assignments: Optional[int] = None,
) -> dict[str, TaskAssignment]:
    """Assign FIFO tasks to the nearest still-idle agent.

    This is the naive MAPD baseline: older tasks are considered first, and each
    task grabs the closest available AGV to its pickup.
    """
    remaining_agents = set(agent_positions)
    out: dict[str, TaskAssignment] = {}
    limit = _assignment_limit(agent_positions, tasks, max_assignments)

    for task in tasks:
        if len(out) >= limit or not remaining_agents:
            break
        best_agent = min(
            remaining_agents,
            key=lambda agent: (
                _safe_distance(router, agent_positions[agent], task.pickup),
                agent,
            ),
        )
        cost = _safe_distance(router, agent_positions[best_agent], task.pickup)
        if not math.isfinite(cost):
            continue
        remaining_agents.remove(best_agent)
        out[best_agent] = TaskAssignment(best_agent, task, cost)
    return out


def hungarian_task_assignment(
    agent_positions: Mapping[str, str],
    tasks: Sequence[Task],
    router: DistanceRouter,
    max_assignments: Optional[int] = None,
) -> dict[str, TaskAssignment]:
    """Globally minimize agent-to-pickup distance with the Hungarian algorithm."""
    return {
        agent: TaskAssignment(agent, tasks[col], cost)
        for agent, col, cost in _hungarian_indices(
            agent_positions,
            [task.pickup for task in tasks],
            router,
            max_assignments=max_assignments,
        )
    }


def fifo_nearest_goal_assignment(
    agent_positions: Mapping[str, str],
    goals: Sequence[str],
    router: DistanceRouter,
    max_assignments: Optional[int] = None,
) -> dict[str, GoalAssignment]:
    """Assign ordered goal nodes FIFO-style to nearest available agents."""
    remaining_agents = set(agent_positions)
    out: dict[str, GoalAssignment] = {}
    limit = _assignment_limit(agent_positions, goals, max_assignments)

    for goal in goals:
        if len(out) >= limit or not remaining_agents:
            break
        best_agent = min(
            remaining_agents,
            key=lambda agent: (
                _safe_distance(router, agent_positions[agent], goal),
                agent,
            ),
        )
        cost = _safe_distance(router, agent_positions[best_agent], goal)
        if not math.isfinite(cost):
            continue
        remaining_agents.remove(best_agent)
        out[best_agent] = GoalAssignment(best_agent, goal, cost)
    return out


def hungarian_goal_assignment(
    agent_positions: Mapping[str, str],
    goals: Sequence[str],
    router: DistanceRouter,
    max_assignments: Optional[int] = None,
) -> dict[str, GoalAssignment]:
    """Globally minimize agent-to-goal distance for one-shot MAPF instances."""
    return {
        agent: GoalAssignment(agent, goals[col], cost)
        for agent, col, cost in _hungarian_indices(
            agent_positions,
            goals,
            router,
            max_assignments=max_assignments,
        )
    }


def _hungarian_indices(
    agent_positions: Mapping[str, str],
    targets: Sequence[str],
    router: DistanceRouter,
    max_assignments: Optional[int],
) -> list[tuple[str, int, float]]:
    agents = list(agent_positions)
    if not agents or not targets:
        return []

    cost = np.full((len(agents), len(targets)), math.inf, dtype=np.float64)
    for i, agent in enumerate(agents):
        source = agent_positions[agent]
        for j, target in enumerate(targets):
            cost[i, j] = _safe_distance(router, source, target)

    finite = cost[np.isfinite(cost)]
    if finite.size == 0:
        return []

    penalty = float(finite.max()) + 1_000_000.0
    safe_cost = np.where(np.isfinite(cost), cost, penalty)
    rows, cols = linear_sum_assignment(safe_cost)

    assignments: list[tuple[str, int, float]] = []
    for row, col in zip(rows, cols):
        original = float(cost[row, col])
        if not math.isfinite(original):
            continue
        assignments.append((agents[row], int(col), original))

    assignments.sort(key=lambda item: (item[2], item[0], item[1]))
    limit = _assignment_limit(agent_positions, targets, max_assignments)
    return assignments[:limit]


def _assignment_limit(
    agent_positions: Mapping[str, str],
    targets: Sequence[object],
    max_assignments: Optional[int],
) -> int:
    natural = min(len(agent_positions), len(targets))
    return natural if max_assignments is None else min(natural, max_assignments)


def _safe_distance(router: DistanceRouter, source: str, target: str) -> float:
    try:
        return float(router.distance(source, target))
    except (KeyError, ValueError):
        return math.inf
