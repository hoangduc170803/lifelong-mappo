"""Naive openTCS-style dispatcher/router/scheduler baseline.

This emulator is intentionally scoped for the Sprint 2 one-shot MAPF table:
  - vehicles are considered in stable name order;
  - each vehicle greedily takes the cheapest remaining goal by routing cost;
  - routes are individual shortest paths over the warehouse graph;
  - a lightweight FIFO-ish resource allocator inserts waits when point/path
    allocations would conflict.

It is not a drop-in implementation of the openTCS kernel. It is a documented,
deterministic baseline for the production control stack shape. Deadlock
recovery/rerouting is deliberately not emulated; cyclic waits therefore time out.
For benchmark tables the honest baseline identifier is ``opentcs_naive``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import networkx as nx

from src.baselines.dispatchers import DistanceRouter, GoalAssignment
from src.mapf.cbs_solver import MAPFPlanResult, validate_paths


def opentcs_default_goal_assignment(
    agent_positions: Mapping[str, str],
    goals: Sequence[str],
    router: DistanceRouter,
) -> dict[str, GoalAssignment]:
    """Vehicle-first greedy assignment by routing cost, then stable goal name."""
    remaining = list(goals)
    assignments: dict[str, GoalAssignment] = {}
    for agent in sorted(agent_positions):
        if not remaining:
            break
        source = agent_positions[agent]
        ranked = sorted(
            (
                (_safe_distance(router, source, goal), str(goal), idx, goal)
                for idx, goal in enumerate(remaining)
            ),
            key=lambda item: (item[0], item[1], item[2]),
        )
        cost, _, idx, goal = ranked[0]
        if not math.isfinite(cost):
            continue
        assignments[agent] = GoalAssignment(agent=agent, goal=goal, cost=cost)
        remaining.pop(idx)
    return assignments


@dataclass(frozen=True)
class OpenTCSRoute:
    """Shortest route tracked for scheduler emulation diagnostics."""

    agent: str
    nodes: tuple[str, ...]


class OpenTCSNaiveEmulator:
    """Emulate OpenTCS-style dispatch + shortest-path routing + queued waits."""

    def __init__(
        self,
        G: nx.DiGraph,
        max_time: int = 512,
        check_following: bool = False,
    ):
        self.G = G
        self.max_time = max_time
        self.check_following = check_following

    def plan(
        self,
        starts: Mapping[str, str],
        goals: Mapping[str, str],
        router,
    ) -> MAPFPlanResult:
        started = time.perf_counter()
        agents = sorted(starts)
        missing = [agent for agent in agents if agent not in goals]
        if missing:
            return MAPFPlanResult(
                success=False,
                solver="opentcs_naive",
                elapsed_s=time.perf_counter() - started,
                diagnostics={"error": "missing assigned goal", "agents": missing},
            )

        routes: dict[str, OpenTCSRoute] = {}
        for agent in agents:
            route = router.path(starts[agent], goals[agent])
            if route is None:
                return MAPFPlanResult(
                    success=False,
                    solver="opentcs_naive",
                    elapsed_s=time.perf_counter() - started,
                    diagnostics={
                        "error": "unroutable assigned goal",
                        "agent": agent,
                        "start": starts[agent],
                        "goal": goals[agent],
                    },
                )
            routes[agent] = OpenTCSRoute(agent=agent, nodes=tuple(route))

        paths = {agent: [starts[agent]] for agent in agents}
        route_idx = {agent: 0 for agent in agents}
        blocked_since: dict[str, Optional[int]] = {agent: None for agent in agents}
        blocked_move_requests = 0
        queued_retries = 0

        for step in range(self.max_time):
            if all(route_idx[a] >= len(routes[a].nodes) - 1 for a in agents):
                break

            current = {agent: routes[agent].nodes[route_idx[agent]] for agent in agents}
            wants = {
                agent: self._desired_next(routes[agent].nodes, route_idx[agent])
                for agent in agents
            }
            moving_out = {
                agent
                for agent, nxt in wants.items()
                if nxt != current[agent]
            }

            planned_next = dict(current)
            accepted_moves: set[str] = set()
            reserved_nodes: dict[str, str] = {
                node: agent
                for agent, node in current.items()
                if agent not in moving_out
            }
            reserved_edges: set[tuple[str, str]] = set()

            request_order = sorted(
                agents,
                key=lambda agent: (
                    blocked_since[agent] if blocked_since[agent] is not None else math.inf,
                    agent,
                ),
            )

            for agent in request_order:
                src = current[agent]
                dst = wants[agent]
                if dst == src:
                    planned_next[agent] = src
                    blocked_since[agent] = None
                    reserved_nodes.setdefault(src, agent)
                    continue

                occupant = next(
                    (
                        other
                        for other, other_node in current.items()
                        if other != agent and other_node == dst
                    ),
                    None,
                )
                occupant_will_leave = (
                    occupant is not None
                    and occupant in accepted_moves
                    and planned_next[occupant] != dst
                )
                node_reserved = dst in reserved_nodes and reserved_nodes[dst] != agent
                edge_reserved = (src, dst) in reserved_edges or (dst, src) in reserved_edges
                destination_occupied = occupant is not None and not occupant_will_leave

                if node_reserved or edge_reserved or destination_occupied:
                    planned_next[agent] = src
                    reserved_nodes.setdefault(src, agent)
                    blocked_move_requests += 1
                    if blocked_since[agent] is not None:
                        queued_retries += 1
                    blocked_since[agent] = (
                        step if blocked_since[agent] is None else blocked_since[agent]
                    )
                    continue

                planned_next[agent] = dst
                accepted_moves.add(agent)
                reserved_nodes[dst] = agent
                reserved_edges.add((src, dst))
                blocked_since[agent] = None

            for agent in agents:
                nxt = planned_next[agent]
                if nxt != current[agent]:
                    route_idx[agent] += 1
                paths[agent].append(nxt)

        completed = all(route_idx[a] >= len(routes[a].nodes) - 1 for a in agents)
        conflicts = validate_paths(self.G, paths, check_following=self.check_following)
        success = completed and not conflicts
        diagnostics = {
            "assignment": "opentcs_naive_vehicle_first_cost_greedy",
            "routing": "individual_shortest_path_distance",
            "scheduler": "queued_resource_allocator_emulator",
            "deadlock_recovery": "not_emulated",
            "blocked_move_requests": blocked_move_requests,
            "queued_retries": queued_retries,
            "routes": {agent: len(route.nodes) - 1 for agent, route in routes.items()},
            "timed_out": not completed,
        }
        if conflicts:
            diagnostics["num_conflicts"] = len(conflicts)

        return MAPFPlanResult(
            success=success,
            paths=paths,
            solver="opentcs_naive",
            makespan=max((len(path) - 1 for path in paths.values()), default=0),
            sum_of_costs=sum(
                1
                for path in paths.values()
                for idx in range(1, len(path))
                if path[idx] != path[idx - 1]
            ),
            conflicts=conflicts,
            elapsed_s=time.perf_counter() - started,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _desired_next(route: Sequence[str], idx: int) -> str:
        if idx >= len(route) - 1:
            return route[-1]
        return route[idx + 1]


def _safe_distance(router: DistanceRouter, source: str, target: str) -> float:
    try:
        return float(router.distance(source, target))
    except (KeyError, ValueError):
        return math.inf
