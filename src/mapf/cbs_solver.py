"""MAPF baseline integration with optional `cbs-mapf` support.

The PyPI package `cbs-mapf` plans on a grid, while the warehouse model is a
directed OpenTCS graph. This module therefore treats the package as an optional
backend: if it is installed, we call it on a compact coordinate grid and then
validate that every returned step is a real directed graph transition. If that
validation fails, or the package is unavailable, we fall back to a graph-native
prioritized space-time A* planner.
"""

from __future__ import annotations

import heapq
import importlib
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import networkx as nx


@dataclass(frozen=True)
class PathConflict:
    """A conflict detected in a multi-agent path set."""

    type: str
    time: int
    agents: tuple[str, str]
    node: Optional[str] = None
    edge: Optional[tuple[str, str]] = None


@dataclass
class MAPFPlanResult:
    """Result returned by MAPF planners."""

    success: bool
    paths: dict[str, list[str]] = field(default_factory=dict)
    solver: str = ""
    makespan: int = 0
    sum_of_costs: int = 0
    conflicts: list[PathConflict] = field(default_factory=list)
    elapsed_s: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _node_at(path: Sequence[str], t: int) -> str:
    return path[t] if t < len(path) else path[-1]


def validate_paths(
    G: nx.DiGraph,
    paths: Mapping[str, Sequence[str]],
    check_following: bool = False,
) -> list[PathConflict]:
    """Return all graph/topology conflicts found in a path dictionary."""
    conflicts: list[PathConflict] = []
    agents = list(paths)
    if not agents:
        return conflicts

    for agent, path in paths.items():
        if not path:
            conflicts.append(PathConflict("empty_path", 0, (agent, agent)))
            continue
        for t, node in enumerate(path):
            if node not in G:
                conflicts.append(PathConflict("unknown_node", t, (agent, agent), node=node))
        for t in range(len(path) - 1):
            u, v = path[t], path[t + 1]
            if u != v and (u not in G or v not in G or not G.has_edge(u, v)):
                conflicts.append(
                    PathConflict("invalid_transition", t + 1, (agent, agent), edge=(u, v))
                )

    horizon = max(len(path) for path in paths.values())
    for t in range(horizon):
        occupied: dict[str, str] = {}
        for agent in agents:
            node = _node_at(paths[agent], t)
            other = occupied.get(node)
            if other is not None:
                conflicts.append(PathConflict("vertex", t, (other, agent), node=node))
            occupied[node] = agent

    for t in range(1, horizon):
        for i, a in enumerate(agents):
            a_prev = _node_at(paths[a], t - 1)
            a_now = _node_at(paths[a], t)
            for b in agents[i + 1 :]:
                b_prev = _node_at(paths[b], t - 1)
                b_now = _node_at(paths[b], t)
                if a_prev == b_now and b_prev == a_now and a_prev != a_now:
                    conflicts.append(PathConflict("edge_swap", t, (a, b), edge=(a_prev, a_now)))
                if check_following:
                    if a_now == b_prev and b_now != b_prev and a_now != b_now:
                        conflicts.append(PathConflict("following", t, (a, b), node=a_now))
                    if b_now == a_prev and a_now != a_prev and b_now != a_now:
                        conflicts.append(PathConflict("following", t, (b, a), node=b_now))

    return conflicts


class PrioritizedGraphPlanner:
    """Graph-native prioritized space-time A* fallback."""

    def __init__(
        self,
        G: nx.DiGraph,
        max_time: int = 512,
        check_following: bool = False,
        reserve_goal_forever: bool = True,
    ):
        self.G = G
        self.max_time = max_time
        self.check_following = check_following
        self.reserve_goal_forever = reserve_goal_forever
        self._reverse = G.reverse(copy=False)

    def plan(
        self,
        starts: Mapping[str, str],
        goals: Mapping[str, str],
        priority_order: Optional[Sequence[str]] = None,
    ) -> MAPFPlanResult:
        start_time = time.perf_counter()
        agents = list(priority_order) if priority_order is not None else list(starts)
        missing = [agent for agent in agents if agent not in starts or agent not in goals]
        if missing:
            return MAPFPlanResult(
                success=False,
                solver="prioritized_graph_astar",
                elapsed_s=time.perf_counter() - start_time,
                diagnostics={"error": "missing start/goal", "agents": missing},
            )

        paths: dict[str, list[str]] = {}
        vertex_reservations: dict[int, set[str]] = defaultdict(set)
        edge_reservations: dict[int, set[tuple[str, str]]] = defaultdict(set)

        for agent in agents:
            path = self._space_time_astar(
                start=starts[agent],
                goal=goals[agent],
                vertex_reservations=vertex_reservations,
                edge_reservations=edge_reservations,
            )
            if path is None:
                return MAPFPlanResult(
                    success=False,
                    paths=paths,
                    solver="prioritized_graph_astar",
                    elapsed_s=time.perf_counter() - start_time,
                    diagnostics={
                        "failed_agent": agent,
                        "failure_subtype": self._failure_subtype(
                            starts[agent],
                            goals[agent],
                        ),
                    },
                )
            paths[agent] = path
            self._reserve_path(path, vertex_reservations, edge_reservations)

        conflicts = validate_paths(self.G, paths, check_following=self.check_following)
        return MAPFPlanResult(
            success=not conflicts,
            paths=paths,
            solver="prioritized_graph_astar",
            makespan=_makespan(paths),
            sum_of_costs=_sum_of_costs(paths),
            conflicts=conflicts,
            elapsed_s=time.perf_counter() - start_time,
        )

    def _heuristic_to_goal(self, goal: str) -> dict[str, int]:
        try:
            return nx.single_source_shortest_path_length(self._reverse, goal)
        except nx.NetworkXError:
            return {}

    def _space_time_astar(
        self,
        start: str,
        goal: str,
        vertex_reservations: Mapping[int, set[str]],
        edge_reservations: Mapping[int, set[tuple[str, str]]],
    ) -> Optional[list[str]]:
        if start not in self.G or goal not in self.G:
            return None
        heuristic = self._heuristic_to_goal(goal)
        if start != goal and start not in heuristic:
            return None
        if start in vertex_reservations.get(0, set()):
            return None

        frontier: list[tuple[int, int, str, int]] = []
        counter = 0
        heapq.heappush(frontier, (heuristic.get(start, 0), counter, start, 0))
        came_from: dict[tuple[str, int], tuple[str, int]] = {}
        best_cost: dict[tuple[str, int], int] = {(start, 0): 0}

        while frontier:
            _, _, node, t = heapq.heappop(frontier)
            if node == goal and self._goal_is_clear(goal, t, vertex_reservations):
                return self._reconstruct_path((node, t), came_from)
            if t >= self.max_time:
                continue

            candidates = [node]
            candidates.extend(self.G.successors(node))
            for nxt in candidates:
                nt = t + 1
                if not self._transition_is_clear(
                    node,
                    nxt,
                    t,
                    nt,
                    vertex_reservations,
                    edge_reservations,
                ):
                    continue
                state = (nxt, nt)
                cost = nt
                if cost >= best_cost.get(state, math.inf):
                    continue
                best_cost[state] = cost
                came_from[state] = (node, t)
                counter += 1
                priority = cost + heuristic.get(nxt, self.max_time)
                heapq.heappush(frontier, (priority, counter, nxt, nt))

        return None

    def _failure_subtype(self, start: str, goal: str) -> str:
        if start not in self.G or goal not in self.G:
            return "no_path"
        heuristic = self._heuristic_to_goal(goal)
        if start != goal and start not in heuristic:
            return "no_path"
        return "prioritized_block"

    def _transition_is_clear(
        self,
        src: str,
        dst: str,
        t: int,
        nt: int,
        vertex_reservations: Mapping[int, set[str]],
        edge_reservations: Mapping[int, set[tuple[str, str]]],
    ) -> bool:
        if dst in vertex_reservations.get(nt, set()):
            return False
        if dst != src and (dst, src) in edge_reservations.get(nt, set()):
            return False
        if self.check_following and dst != src and dst in vertex_reservations.get(t, set()):
            return False
        return True

    def _goal_is_clear(
        self,
        goal: str,
        arrival_time: int,
        vertex_reservations: Mapping[int, set[str]],
    ) -> bool:
        for t in range(arrival_time, self.max_time + 1):
            if goal in vertex_reservations.get(t, set()):
                return False
        return True

    def _reserve_path(
        self,
        path: Sequence[str],
        vertex_reservations: dict[int, set[str]],
        edge_reservations: dict[int, set[tuple[str, str]]],
    ) -> None:
        for t, node in enumerate(path):
            vertex_reservations[t].add(node)
            if t > 0:
                edge_reservations[t].add((path[t - 1], node))
        if not self.reserve_goal_forever:
            return
        goal = path[-1]
        for t in range(len(path), self.max_time + 1):
            vertex_reservations[t].add(goal)

    @staticmethod
    def _reconstruct_path(
        state: tuple[str, int],
        came_from: Mapping[tuple[str, int], tuple[str, int]],
    ) -> list[str]:
        path = [state[0]]
        while state in came_from:
            state = came_from[state]
            path.append(state[0])
        path.reverse()
        return path


class CBSMapfPlanner:
    """Adapter around `cbs-mapf` with graph-safe fallback."""

    def __init__(
        self,
        G: nx.DiGraph,
        max_time: int = 512,
        robot_radius: int = 0,
        low_level_max_iter: int = 100,
        max_iter: int = 200,
        max_process: int = 1,
        check_following: bool = False,
    ):
        self.G = G
        self.max_time = max_time
        self.robot_radius = robot_radius
        self.low_level_max_iter = low_level_max_iter
        self.max_iter = max_iter
        self.max_process = max_process
        self.check_following = check_following
        self._coord_offset = 1000
        self._coord_collision_nodes: list[dict[str, Any]] = []
        self.fallback = PrioritizedGraphPlanner(
            G,
            max_time=max_time,
            check_following=check_following,
        )
        self._node_to_coord, self._coord_to_node, self._static_obstacles = (
            self._build_compact_grid()
        )

    def compact_grid_diagnostics(self) -> dict[str, Any]:
        """Diagnostics for the lossy graph-to-grid projection used by cbs-mapf."""
        return {
            "graph_nodes": self.G.number_of_nodes(),
            "unique_compact_coords": len(self._node_to_coord),
            "coord_collision_nodes": len(self._coord_collision_nodes),
            "coord_collision_samples": self._coord_collision_nodes[:10],
            "compact_grid_obstacles": len(self._static_obstacles),
        }

    def plan(
        self,
        starts: Mapping[str, str],
        goals: Mapping[str, str],
        use_external: bool = True,
        fallback_on_failure: bool = True,
    ) -> MAPFPlanResult:
        if use_external:
            external = self._plan_with_cbs_mapf(starts, goals)
            if external.success:
                return external
            if not fallback_on_failure:
                return external
            fallback = self.fallback.plan(starts, goals)
            fallback.diagnostics["external_cbs_mapf"] = external.diagnostics
            return fallback
        return self.fallback.plan(starts, goals)

    def _build_compact_grid(
        self,
    ) -> tuple[dict[str, tuple[int, int]], dict[tuple[int, int], str], list[tuple[int, int]]]:
        xs = sorted({float(data["x"]) for _, data in self.G.nodes(data=True)})
        ys = sorted({float(data["y"]) for _, data in self.G.nodes(data=True)})
        x_to_idx = {x: i + self._coord_offset + 1 for i, x in enumerate(xs)}
        y_to_idx = {y: i + self._coord_offset + 1 for i, y in enumerate(ys)}

        node_to_coord: dict[str, tuple[int, int]] = {}
        coord_to_node: dict[tuple[int, int], str] = {}
        self._coord_collision_nodes.clear()
        for node, data in self.G.nodes(data=True):
            coord = (x_to_idx[float(data["x"])], y_to_idx[float(data["y"])])
            if coord in coord_to_node:
                self._coord_collision_nodes.append(
                    {
                        "node": node,
                        "kept_node": coord_to_node[coord],
                        "coord": coord,
                    }
                )
                continue
            node_to_coord[node] = coord
            coord_to_node[coord] = node

        occupied = set(coord_to_node)
        x_min = self._coord_offset
        x_max = self._coord_offset + len(xs) + 1
        y_min = self._coord_offset
        y_max = self._coord_offset + len(ys) + 1

        static_obstacles = []
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                is_border = x in (x_min, x_max) or y in (y_min, y_max)
                is_internal_obstacle = not is_border and (x, y) not in occupied
                if is_border or is_internal_obstacle:
                    static_obstacles.append((x, y))
        return node_to_coord, coord_to_node, static_obstacles

    def _plan_with_cbs_mapf(
        self,
        starts: Mapping[str, str],
        goals: Mapping[str, str],
    ) -> MAPFPlanResult:
        start_time = time.perf_counter()
        agents = list(starts)
        try:
            start_coords = [self._node_to_coord[starts[agent]] for agent in agents]
            goal_coords = [self._node_to_coord[goals[agent]] for agent in agents]
        except KeyError as exc:
            return MAPFPlanResult(
                success=False,
                solver="cbs_mapf",
                elapsed_s=time.perf_counter() - start_time,
                diagnostics={
                    "error": "node has no unique compact coordinate",
                    "node": str(exc),
                    **self.compact_grid_diagnostics(),
                },
            )

        try:
            planner_mod = importlib.import_module("cbs_mapf.planner")
            agent_mod = importlib.import_module("cbs_mapf.agent")
        except ImportError as exc:
            return MAPFPlanResult(
                success=False,
                solver="cbs_mapf",
                elapsed_s=time.perf_counter() - start_time,
                diagnostics={
                    "error": "cbs-mapf package not installed",
                    "exception": str(exc),
                    **self.compact_grid_diagnostics(),
                },
            )

        try:
            planner = planner_mod.Planner(
                grid_size=1,
                robot_radius=self.robot_radius,
                static_obstacles=self._static_obstacles,
            )

            def keep_agent_goal_pairs(starts_arg, goals_arg):
                return [
                    agent_mod.Agent(start, goal)
                    for start, goal in zip(starts_arg, goals_arg)
                ]

            raw_paths = planner.plan(
                start_coords,
                goal_coords,
                assign=keep_agent_goal_pairs,
                max_iter=self.max_iter,
                low_level_max_iter=self.low_level_max_iter,
                max_process=self.max_process,
                debug=False,
            )
        except Exception as exc:  # pragma: no cover - depends on optional package
            return MAPFPlanResult(
                success=False,
                solver="cbs_mapf",
                elapsed_s=time.perf_counter() - start_time,
                diagnostics={
                    "error": "cbs-mapf planning failed",
                    "exception": repr(exc),
                    **self.compact_grid_diagnostics(),
                },
            )

        if raw_paths is None or len(raw_paths) == 0:
            return MAPFPlanResult(
                success=False,
                solver="cbs_mapf",
                elapsed_s=time.perf_counter() - start_time,
                diagnostics={
                    "error": "cbs-mapf returned no paths",
                    **self.compact_grid_diagnostics(),
                },
            )

        paths = self._decode_external_paths(agents, raw_paths)
        conflicts = validate_paths(self.G, paths, check_following=self.check_following)
        if conflicts:
            return MAPFPlanResult(
                success=False,
                paths=paths,
                solver="cbs_mapf",
                conflicts=conflicts,
                elapsed_s=time.perf_counter() - start_time,
                diagnostics={
                    "error": "cbs-mapf returned paths that do not respect the directed graph",
                    "num_conflicts": len(conflicts),
                    **self.compact_grid_diagnostics(),
                },
            )

        return MAPFPlanResult(
            success=True,
            paths=paths,
            solver="cbs_mapf",
            makespan=_makespan(paths),
            sum_of_costs=_sum_of_costs(paths),
            elapsed_s=time.perf_counter() - start_time,
            diagnostics=self.compact_grid_diagnostics(),
        )

    def _decode_external_paths(self, agents: Sequence[str], raw_paths: Any) -> dict[str, list[str]]:
        paths: dict[str, list[str]] = {}
        for idx, agent in enumerate(agents):
            decoded: list[str] = []
            for coord_like in raw_paths[idx]:
                coord = tuple(int(round(float(v))) for v in coord_like[:2])
                node = self._coord_to_node.get(coord)
                decoded.append(node if node is not None else f"<unknown:{coord[0]},{coord[1]}>")
            paths[agent] = _trim_repeated_tail(decoded)
        return paths


def _trim_repeated_tail(path: list[str]) -> list[str]:
    while len(path) > 1 and path[-1] == path[-2]:
        path.pop()
    return path


def _makespan(paths: Mapping[str, Sequence[str]]) -> int:
    return max((len(path) - 1 for path in paths.values()), default=0)


def _sum_of_costs(paths: Mapping[str, Sequence[str]]) -> int:
    total = 0
    for path in paths.values():
        total += sum(1 for i in range(1, len(path)) if path[i] != path[i - 1])
    return total
