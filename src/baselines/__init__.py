"""Classical Sprint 2 baselines for assignment and MAPF benchmarking."""

from .dispatchers import (
    GoalAssignment,
    TaskAssignment,
    fifo_nearest_goal_assignment,
    fifo_nearest_task_assignment,
    hungarian_goal_assignment,
    hungarian_task_assignment,
)
from .opentcs_default import (
    OpenTCSNaiveEmulator,
    opentcs_default_goal_assignment,
)

__all__ = [
    "GoalAssignment",
    "OpenTCSNaiveEmulator",
    "TaskAssignment",
    "fifo_nearest_goal_assignment",
    "fifo_nearest_task_assignment",
    "hungarian_goal_assignment",
    "hungarian_task_assignment",
    "opentcs_default_goal_assignment",
]
