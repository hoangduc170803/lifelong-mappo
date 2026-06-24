"""Sprint 2 classical baseline benchmark runner.

The runner creates one-shot MAPF instances on the largest connected warehouse
component, runs classical assignment/planning baselines, and writes paper-ready
CSV rows with makespan, waiting time, conflicts, success, and timing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import networkx as nx
import numpy as np

from src.baselines.dispatchers import (
    fifo_nearest_goal_assignment,
    hungarian_goal_assignment,
)
from src.baselines.opentcs_default import (
    OpenTCSNaiveEmulator,
    opentcs_default_goal_assignment,
)
from src.map_parser import parse_opentcs_map
from src.mapf.cbs_solver import CBSMapfPlanner, MAPFPlanResult
from src.mapf.priority_search import PrioritySearchPlanner
from src.routing.astar import AStarRouter


DEFAULT_MAP_FILE = (
    Path(__file__).resolve().parents[2]
    / "warehouse_map.xml"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "results" / "baselines" / "sprint2_mapf_baselines.csv"
DEFAULT_BASELINES = (
    "prioritized_planning_default_order",
    "priority_search",
    "hungarian_prioritized_planning",
    "fifo_nearest_prioritized_planning",
    "opentcs_naive",
)


@dataclass(frozen=True)
class BenchmarkRow:
    """One CSV row for a baseline/seed/agent-count run."""

    baseline: str
    seed: int
    num_agents: int
    success: bool
    success_rate: float
    solver: str
    failure_reason: str
    instance_makespan: int
    lower_bound_steps: int
    lower_bound_distance: float
    makespan_over_lower_bound: float
    sum_of_costs: int
    waiting_time_agent_steps: int
    throughput_tasks_per_1000_steps: float
    conflicts_total: int
    elapsed_assignment_s: float
    elapsed_planner_s: float
    elapsed_s: float
    starts_json: str
    goals_json: str
    diagnostics_json: str


@dataclass(frozen=True)
class BaselineRun:
    """Planner result plus benchmark timing metadata."""

    result: MAPFPlanResult
    goals: dict[str, str]
    elapsed_assignment_s: float = 0.0
    elapsed_planner_s: float = 0.0

    @property
    def elapsed_s(self) -> float:
        return self.elapsed_assignment_s + self.elapsed_planner_s


def sample_mapf_instance(
    G: nx.DiGraph,
    num_agents: int,
    seed: int,
) -> tuple[dict[str, str], dict[str, str]]:
    """Sample distinct starts and distinct direct goals for one-shot MAPF."""
    if num_agents < 1:
        raise ValueError("num_agents must be >= 1")
    nodes = np.array(list(G.nodes()), dtype=object)
    if len(nodes) < 2 * num_agents:
        raise ValueError("graph needs at least 2 * num_agents nodes")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(nodes, size=2 * num_agents, replace=False)
    agents = [f"agv_{i}" for i in range(num_agents)]
    starts = {agent: str(node) for agent, node in zip(agents, chosen[:num_agents])}
    goals = {agent: str(node) for agent, node in zip(agents, chosen[num_agents:])}
    return starts, goals


def run_mapf_baseline(
    G: nx.DiGraph,
    starts: Mapping[str, str],
    goals: Mapping[str, str],
    baseline: str,
    seed: int = 0,
    max_time: int = 512,
    use_external: bool = True,
    router: AStarRouter | None = None,
) -> BaselineRun:
    """Run one Sprint 2 baseline and return result plus assigned goals."""
    baseline_key = baseline.lower()
    assigned_goals = dict(goals)

    if baseline_key in {"hungarian_prioritized_planning", "hungarian_pp"}:
        assignment_router = router if router is not None else AStarRouter(G, precompute=True)
        assignment_start = time.perf_counter()
        assignments = hungarian_goal_assignment(starts, list(goals.values()), assignment_router)
        elapsed_assignment_s = time.perf_counter() - assignment_start
        assigned_goals = {agent: item.goal for agent, item in assignments.items()}
        planner_start = time.perf_counter()
        result = CBSMapfPlanner(G, max_time=max_time).plan(
            starts,
            assigned_goals,
            use_external=False,
        )
        elapsed_planner_s = time.perf_counter() - planner_start
        result.diagnostics["assignment"] = "hungarian_goal"
        result.diagnostics["assigned_agents"] = len(assigned_goals)
        result.diagnostics["external_backend_enabled"] = False
        result.diagnostics["baseline_note"] = "Hungarian assignment + graph-native prioritized planning"
        return BaselineRun(result, assigned_goals, elapsed_assignment_s, elapsed_planner_s)

    if baseline_key in {"hungarian_cbs", "hungarian"}:
        assignment_router = router if router is not None else AStarRouter(G, precompute=True)
        assignment_start = time.perf_counter()
        assignments = hungarian_goal_assignment(starts, list(goals.values()), assignment_router)
        elapsed_assignment_s = time.perf_counter() - assignment_start
        assigned_goals = {agent: item.goal for agent, item in assignments.items()}
        planner_start = time.perf_counter()
        result = CBSMapfPlanner(G, max_time=max_time).plan(
            starts,
            assigned_goals,
            use_external=use_external,
        )
        elapsed_planner_s = time.perf_counter() - planner_start
        result.diagnostics["assignment"] = "hungarian_goal"
        result.diagnostics["assigned_agents"] = len(assigned_goals)
        result.diagnostics["external_backend_enabled"] = use_external
        return BaselineRun(result, assigned_goals, elapsed_assignment_s, elapsed_planner_s)

    if baseline_key in {"fifo_nearest_prioritized_planning", "fifo_nearest_pp"}:
        assignment_router = router if router is not None else AStarRouter(G, precompute=True)
        assignment_start = time.perf_counter()
        assignments = fifo_nearest_goal_assignment(starts, list(goals.values()), assignment_router)
        elapsed_assignment_s = time.perf_counter() - assignment_start
        assigned_goals = {agent: item.goal for agent, item in assignments.items()}
        planner_start = time.perf_counter()
        result = CBSMapfPlanner(G, max_time=max_time).plan(
            starts,
            assigned_goals,
            use_external=False,
        )
        elapsed_planner_s = time.perf_counter() - planner_start
        result.diagnostics["assignment"] = "fifo_nearest_goal"
        result.diagnostics["assigned_agents"] = len(assigned_goals)
        result.diagnostics["external_backend_enabled"] = False
        result.diagnostics["baseline_note"] = "FIFO-nearest assignment + graph-native prioritized planning"
        return BaselineRun(result, assigned_goals, elapsed_assignment_s, elapsed_planner_s)

    if baseline_key in {"fifo_nearest", "fifo_nearest_cbs"}:
        assignment_router = router if router is not None else AStarRouter(G, precompute=True)
        assignment_start = time.perf_counter()
        assignments = fifo_nearest_goal_assignment(starts, list(goals.values()), assignment_router)
        elapsed_assignment_s = time.perf_counter() - assignment_start
        assigned_goals = {agent: item.goal for agent, item in assignments.items()}
        planner_start = time.perf_counter()
        result = CBSMapfPlanner(G, max_time=max_time).plan(
            starts,
            assigned_goals,
            use_external=use_external,
        )
        elapsed_planner_s = time.perf_counter() - planner_start
        result.diagnostics["assignment"] = "fifo_nearest_goal"
        result.diagnostics["assigned_agents"] = len(assigned_goals)
        result.diagnostics["external_backend_enabled"] = use_external
        return BaselineRun(result, assigned_goals, elapsed_assignment_s, elapsed_planner_s)

    if baseline_key == "opentcs_naive":
        assignment_router = router if router is not None else AStarRouter(G, precompute=True)
        assignment_start = time.perf_counter()
        assignments = opentcs_default_goal_assignment(
            starts,
            list(goals.values()),
            assignment_router,
        )
        elapsed_assignment_s = time.perf_counter() - assignment_start
        assigned_goals = {agent: item.goal for agent, item in assignments.items()}
        planner_start = time.perf_counter()
        result = OpenTCSNaiveEmulator(G, max_time=max_time).plan(
            starts,
            assigned_goals,
            assignment_router,
        )
        elapsed_planner_s = time.perf_counter() - planner_start
        result.diagnostics["assigned_agents"] = len(assigned_goals)
        result.diagnostics["emulation_scope"] = "dispatcher_router_scheduler"
        return BaselineRun(result, assigned_goals, elapsed_assignment_s, elapsed_planner_s)

    if baseline_key in {"pbs", "priority_search"}:
        planner_start = time.perf_counter()
        result = PrioritySearchPlanner(G, max_time=max_time).plan(
            starts,
            assigned_goals,
            seed=seed,
        )
        elapsed_planner_s = time.perf_counter() - planner_start
        result.diagnostics["baseline_note"] = (
            "multi-order prioritized planning; full PBS constraint tree deferred"
        )
        return BaselineRun(result, assigned_goals, 0.0, elapsed_planner_s)

    if baseline_key in {"prioritized_planning_default_order", "prioritized_planning"}:
        planner_start = time.perf_counter()
        result = CBSMapfPlanner(G, max_time=max_time).plan(
            starts,
            assigned_goals,
            use_external=False,
        )
        elapsed_planner_s = time.perf_counter() - planner_start
        result.diagnostics["assignment"] = "fixed_agent_goal_pairs"
        result.diagnostics["external_backend_enabled"] = False
        result.diagnostics["baseline_note"] = "graph-native prioritized planning in default agent order"
        return BaselineRun(result, assigned_goals, 0.0, elapsed_planner_s)

    if baseline_key in {"cbs", "direct_cbs", "cbs_mapf"}:
        planner_start = time.perf_counter()
        result = CBSMapfPlanner(G, max_time=max_time).plan(
            starts,
            assigned_goals,
            use_external=use_external,
        )
        elapsed_planner_s = time.perf_counter() - planner_start
        result.diagnostics["assignment"] = "fixed_agent_goal_pairs"
        result.diagnostics["external_backend_enabled"] = use_external
        return BaselineRun(result, assigned_goals, 0.0, elapsed_planner_s)

    raise ValueError(f"unknown baseline: {baseline}")


def run_benchmark(
    G: nx.DiGraph,
    agent_counts: Sequence[int] = (10, 15, 20),
    seeds: Iterable[int] = range(5),
    baselines: Sequence[str] = DEFAULT_BASELINES,
    max_time: int = 512,
    use_external: bool = True,
) -> list[BenchmarkRow]:
    """Run a grid of baseline/agent-count/seed experiments."""
    rows: list[BenchmarkRow] = []
    router = AStarRouter(G, precompute=True)
    for num_agents in agent_counts:
        for seed in seeds:
            starts, direct_goals = sample_mapf_instance(G, num_agents, seed)
            for baseline in baselines:
                run = run_mapf_baseline(
                    G,
                    starts,
                    direct_goals,
                    baseline=baseline,
                    seed=seed,
                    max_time=max_time,
                    use_external=use_external,
                    router=router,
                )
                rows.append(
                    benchmark_row(
                        baseline=baseline,
                        seed=seed,
                        num_agents=num_agents,
                        starts=starts,
                        goals=run.goals,
                        run=run,
                        router=router,
                    )
                )
    return rows


def benchmark_row(
    baseline: str,
    seed: int,
    num_agents: int,
    starts: Mapping[str, str],
    goals: Mapping[str, str],
    run: BaselineRun,
    router: AStarRouter | None = None,
) -> BenchmarkRow:
    """Convert a planner result into a stable CSV row."""
    result = run.result
    completed = num_agents if result.success else 0
    horizon = max(result.makespan, 1)
    throughput = completed / horizon * 1000.0
    lower_bound_steps, lower_bound_distance = _lower_bounds(starts, goals, router)
    ratio = (
        result.makespan / lower_bound_steps
        if result.success and lower_bound_steps > 0
        else math.inf
    )
    diagnostics = _json_dumpable(result.diagnostics)
    return BenchmarkRow(
        baseline=baseline,
        seed=seed,
        num_agents=num_agents,
        success=result.success,
        success_rate=1.0 if result.success else 0.0,
        solver=result.solver,
        failure_reason=_failure_reason(result),
        instance_makespan=result.makespan,
        lower_bound_steps=lower_bound_steps,
        lower_bound_distance=lower_bound_distance,
        makespan_over_lower_bound=ratio,
        sum_of_costs=result.sum_of_costs,
        waiting_time_agent_steps=_waiting_time_agent_steps(result.paths, result.makespan),
        throughput_tasks_per_1000_steps=throughput,
        conflicts_total=len(result.conflicts),
        elapsed_assignment_s=run.elapsed_assignment_s,
        elapsed_planner_s=run.elapsed_planner_s,
        elapsed_s=run.elapsed_s,
        starts_json=json.dumps(dict(starts), sort_keys=True),
        goals_json=json.dumps(dict(goals), sort_keys=True),
        diagnostics_json=json.dumps(diagnostics, sort_keys=True),
    )


def write_benchmark_csv(rows: Sequence[BenchmarkRow], path: Path | str) -> Path:
    """Write benchmark rows to CSV, creating parent folders as needed."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(BenchmarkRow.__dataclass_fields__)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return output


def _waiting_time_agent_steps(
    paths: Mapping[str, Sequence[str]],
    makespan: int,
) -> int:
    total = 0
    for path in paths.values():
        total += sum(1 for i in range(1, len(path)) if path[i] == path[i - 1])
        total += max(0, makespan - (len(path) - 1))
    return total


def _lower_bounds(
    starts: Mapping[str, str],
    goals: Mapping[str, str],
    router: AStarRouter | None,
) -> tuple[int, float]:
    if router is None:
        return 0, 0.0

    max_steps = 0
    max_distance = 0.0
    for agent, start in starts.items():
        goal = goals.get(agent)
        if goal is None:
            continue
        path = router.path(start, goal)
        if path is None:
            return 0, math.inf
        max_steps = max(max_steps, len(path) - 1)
        max_distance = max(max_distance, router.distance(start, goal))
    return max_steps, max_distance


def _failure_reason(result: MAPFPlanResult) -> str:
    if result.success:
        return ""
    diagnostics = result.diagnostics
    failure_subtype = diagnostics.get("failure_subtype")
    if failure_subtype in {"prioritized_block", "no_path"}:
        return str(failure_subtype)
    if diagnostics.get("timed_out"):
        return "timeout"
    if result.conflicts:
        return "conflict"
    error = str(diagnostics.get("error", "")).lower()
    if "unroutable" in error or "no paths" in error or "no unique compact coordinate" in error:
        return "unreachable"
    return "planner_failure"


def _json_dumpable(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _json_dumpable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_dumpable(v) for v in value]
        return repr(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--agents", type=int, nargs="+", default=[10, 15, 20])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--baselines", nargs="+", default=list(DEFAULT_BASELINES))
    parser.add_argument("--max-time", type=int, default=512)
    parser.add_argument(
        "--no-external",
        action="store_true",
        help="Disable optional cbs-mapf backend and use graph-native fallback.",
    )
    args = parser.parse_args(argv)

    G = parse_opentcs_map(str(args.map), restrict_to_largest_scc=True)
    rows = run_benchmark(
        G,
        agent_counts=args.agents,
        seeds=args.seeds,
        baselines=args.baselines,
        max_time=args.max_time,
        use_external=not args.no_external,
    )
    output = write_benchmark_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
