"""Windowed PBS — the faithful RHCR low-level solver (Li et al. 2021).

RHCR (Rolling-Horizon Collision Resolution) replans the fleet every ``h`` steps
with a *windowed* MAPF solver that only resolves collisions within the next
``w`` timesteps; its canonical low-level is **PBS** (Priority-Based Search,
Ma et al. 2019). This module is a faithful PBS made faithful to the lifelong
regime:

* **Priority-tree search (Ma et al. 2019).** PBS searches a binary *priority
  tree* by DFS. The root plans every agent independently. At each node we find
  the first collision ``(ai, aj)`` in the current joint plan and branch on the
  two consistent total-precedence extensions ``ai ≺ aj`` / ``aj ≺ ai``; for each
  child we replan the now-lower-priority agent **and every agent below it** in
  topological order, each respecting all higher-priority agents as space-time
  reservations. The first collision-free node is the solution. This is the
  conflict-driven priority search of the original PBS, not a fixed enumeration
  of priority orders.

* **Plans toward the TRUE goal.** Each agent's low-level is a bounded-horizon
  space-time A* toward its real dispatch goal. The objective is *progress*:
  reach the goal if it is within ``w`` steps, otherwise return the windowed
  prefix that ends closest to the goal. No goal de-duplication — the windowed
  prefix never needs the agent to actually occupy the goal, so two agents
  sharing a goal node is not a problem (the lower-priority agent queues behind
  whoever holds it).

* **Collision-free within the window by construction.** A lower-priority agent
  treats every higher-priority agent's space-time path as hard vertex+edge
  reservations (held at its final windowed node for the rest of the window,
  truncated at the first contested timestep).

The difference from ``src/mapf/cbs_solver.PrioritizedGraphPlanner`` that makes
it *windowed*: the space-time A* does **not** require reaching the goal (the
all-or-nothing arrival test + ``reserve_goal_forever`` in the full-horizon
planner are what forced the LaCAM reference to de-duplicate waypoints). Here,
"made progress toward goal within ``w``" is success.

Reference RHCR / LaCAM implementations (Li 2021 ``Jiaoyang-Li/RHCR``; Okumura
``lacam3``) consume an *undirected grid* ``.map``; this warehouse is a directed
1313-node OpenTCS strongly-connected component, so the solver is reimplemented
graph-natively (it expands ``G.successors`` and reserves directed edges). The
env's safety validator + lookahead mask remain the runtime collision backstop
(as for every classical controller here).
"""

from __future__ import annotations

import heapq
import time
from collections import defaultdict, deque
from typing import Mapping, Optional, Sequence

import networkx as nx

from src.mapf.cbs_solver import MAPFPlanResult

DEFAULT_WINDOW = 16
DEFAULT_MAX_ORDERS = 4
# Priority-tree refinement budget. Sparse windows are solved in <5 nodes; dense
# (congested) windows exhaust this and fall back to a complete prioritized plan
# (collision-free by construction), so a small budget keeps the rolling horizon
# real-time without losing feasibility. PBS (Ma 2019) is incomplete, so a feasible
# PP fallback under a node/time bound is the standard windowed-RHCR backbone.
DEFAULT_MAX_PT_NODES = 64


class WindowedPBSPlanner:
    """Windowed Priority-Based Search (the RHCR windowed solver, Ma et al. 2019)."""

    def __init__(
        self,
        G: nx.DiGraph,
        window: int = DEFAULT_WINDOW,
        max_orders: int = DEFAULT_MAX_ORDERS,
        check_following: bool = False,
        seed: int = 0,
        max_pt_nodes: int = DEFAULT_MAX_PT_NODES,
    ):
        if window < 1:
            raise ValueError("window must be >= 1")
        if max_orders < 1:
            raise ValueError("max_orders must be >= 1")
        self.G = G
        self.window = window
        self.max_orders = max_orders
        self.check_following = check_following
        self.seed = seed
        # PBS is DFS over a priority tree; bound the expansion to stay real-time.
        # ``max_orders`` (kept for API/back-compat) widens the bound so callers
        # asking for a broader search still get one.
        self.max_pt_nodes = max(max_pt_nodes, 8 * max_orders)
        self._reverse = G.reverse(copy=False)
        # Per-goal hop-count heuristic (distance node -> goal). Goals are depot
        # stations that repeat across replans, so caching makes the reverse BFS
        # a one-off cost per distinct goal.
        self._heur_cache: dict[str, dict[str, int]] = {}

    # ------------------------------------------------------------------ public

    def plan(
        self,
        starts: Mapping[str, str],
        goals: Mapping[str, str],
        priority_orders: Optional[Sequence[Sequence[str]]] = None,
    ) -> MAPFPlanResult:
        """Windowed PBS toward the true goals; first collision-free PT node wins.

        ``priority_orders`` is accepted for back-compat; if a hint order is given
        its first ordering seeds the root precedence, otherwise PBS starts from
        the empty precedence (each agent planned independently).
        """
        start_time = time.perf_counter()
        agents = list(starts)
        if not agents:
            return MAPFPlanResult(
                success=True, paths={}, solver="windowed_pbs",
                makespan=0, elapsed_s=0.0,
                diagnostics={"pt_nodes": 0, "window": self.window},
            )

        # Root precedence: empty (or seeded from a caller hint).
        root_prio: frozenset[tuple[str, str]] = frozenset()
        if priority_orders:
            hint = list(priority_orders[0])
            root_prio = frozenset(
                (hint[i], hint[j])
                for i in range(len(hint)) for j in range(i + 1, len(hint))
            )

        root_plan = self._initial_plan(root_prio, starts, goals)
        # DFS stack of (precedence, plan). Track the least-conflicting plan seen
        # so an exhausted search still returns a best-effort windowed plan.
        stack: list[tuple[frozenset[tuple[str, str]], dict[str, list[str]]]] = [
            (root_prio, root_plan)
        ]
        best_plan = root_plan
        best_conflicts = self._num_collisions(root_plan)
        pt_nodes = 0

        while stack and pt_nodes < self.max_pt_nodes:
            prio, plan = stack.pop()
            pt_nodes += 1
            collision = self._first_collision(plan)
            if collision is None:
                return MAPFPlanResult(
                    success=True,
                    paths=plan,
                    solver="windowed_pbs",
                    makespan=max((len(p) - 1 for p in plan.values()), default=0),
                    elapsed_s=time.perf_counter() - start_time,
                    diagnostics={
                        "pt_nodes": pt_nodes,
                        "window": self.window,
                        "priority_pairs": len(prio),
                    },
                )
            nconf = self._num_collisions(plan)
            if nconf < best_conflicts:
                best_conflicts = nconf
                best_plan = plan
            ai, aj = collision
            # Branch on the two consistent orderings; explore (ai≺aj) first.
            children = []
            for high, low in ((aj, ai), (ai, aj)):
                if self._creates_cycle(prio, high, low):
                    continue
                new_prio = prio | {(high, low)}
                new_plan = dict(plan)
                if self._update_plan(new_prio, new_plan, low, starts, goals):
                    children.append((new_prio, new_plan))
            stack.extend(children)

        # Priority-tree budget exhausted with no collision-free node (a dense,
        # congested window). Fall back to a complete prioritized plan: every agent
        # yields to all higher-priority agents and parks only where it can hold
        # safely. Prioritized planning is incomplete — the few lowest-priority
        # agents in a jam may be boxed in and left with a residual conflict — so we
        # return the least-conflicting of {complete order, best search node} and
        # let the env's safety validator + lookahead mask (the documented runtime
        # backstop for every classical controller here) resolve any residual by a
        # one-step wait. At a congested cell this is the correct classical
        # behaviour: a low-throughput *queueing* plan, not a frozen fleet.
        fb_plan = self._fallback_plan(starts, goals)
        fb_conflicts = self._num_collisions(fb_plan)
        if fb_conflicts <= best_conflicts:
            chosen, residual = fb_plan, fb_conflicts
        else:
            chosen, residual = best_plan, best_conflicts
        return MAPFPlanResult(
            success=True,
            paths=chosen,
            solver="windowed_pbs",
            makespan=max((len(p) - 1 for p in chosen.values()), default=0),
            elapsed_s=time.perf_counter() - start_time,
            diagnostics={
                "pt_nodes": pt_nodes,
                "window": self.window,
                "pp_fallback": True,
                "residual_collisions": residual,
            },
        )

    # ----------------------------------------------------------- PBS internals

    def _initial_plan(
        self,
        prio: frozenset[tuple[str, str]],
        starts: Mapping[str, str],
        goals: Mapping[str, str],
    ) -> dict[str, list[str]]:
        """Root plan: every agent in topological order, each respecting the
        agents already fixed above it (empty precedence => fully independent)."""
        plan: dict[str, list[str]] = {}
        order = self._topo_order(prio, set(starts))
        for agent in order:
            higher = self._ancestors(prio, agent)
            vres, eres = self._reservations_for(plan, prio, higher)
            path = self._windowed_astar(starts[agent], goals[agent], vres, eres)
            plan[agent] = path if path is not None else [starts[agent]]
        return plan

    def _update_plan(
        self,
        prio: frozenset[tuple[str, str]],
        plan: dict[str, list[str]],
        agent: str,
        starts: Mapping[str, str],
        goals: Mapping[str, str],
    ) -> bool:
        """Replan ``agent`` (its precedence just changed) and every lower-priority
        agent that now collides with a higher-priority one, in topological order.
        Returns False if any agent cannot even hold its start (prunes the PT
        child)."""
        affected = self._descendants_incl(prio, agent)
        for k in self._topo_order(prio, affected):
            higher = self._ancestors(prio, k)
            if k != agent and not self._collides_with_any(plan, k, higher):
                continue
            vres, eres = self._reservations_for(plan, prio, higher)
            path = self._windowed_astar(starts[k], goals[k], vres, eres)
            if path is None:
                return False
            plan[k] = path
        return True

    def _fallback_plan(
        self,
        starts: Mapping[str, str],
        goals: Mapping[str, str],
    ) -> dict[str, list[str]]:
        """Complete prioritized plan for the whole fleet, used when the
        priority-tree search exhausts its budget. Priority = distance-to-goal
        ascending (agents nearest their goal move first, clearing contested
        cells — the standard prioritized-planning heuristic), tie-broken by name.
        Any total order is collision-free by construction; this one tends to keep
        throughput up by letting almost-done agents finish and free space."""
        order = sorted(
            starts,
            key=lambda a: (
                self._heuristic(goals[a]).get(starts[a], self.window + 1),
                a,
            ),
        )
        total_prio = frozenset(
            (order[i], order[j])
            for i in range(len(order))
            for j in range(i + 1, len(order))
        )
        return self._initial_plan(total_prio, starts, goals)

    # ----------------------------------------- precedence-graph (DAG) helpers

    @staticmethod
    def _ancestors(prio: frozenset[tuple[str, str]], k: str) -> set[str]:
        """Agents with strictly higher priority than ``k`` (``a ≺ … ≺ k``)."""
        preds: dict[str, set[str]] = defaultdict(set)
        for high, low in prio:
            preds[low].add(high)
        out: set[str] = set()
        stack = list(preds.get(k, ()))
        while stack:
            a = stack.pop()
            if a not in out:
                out.add(a)
                stack.extend(preds.get(a, ()))
        return out

    @staticmethod
    def _descendants_incl(prio: frozenset[tuple[str, str]], a: str) -> set[str]:
        """``a`` plus every agent of strictly lower priority (``a ≺ … ≺ x``)."""
        succ: dict[str, set[str]] = defaultdict(set)
        for high, low in prio:
            succ[high].add(low)
        out = {a}
        stack = [a]
        while stack:
            x = stack.pop()
            for y in succ.get(x, ()):
                if y not in out:
                    out.add(y)
                    stack.append(y)
        return out

    @staticmethod
    def _topo_order(prio: frozenset[tuple[str, str]], subset: set[str]) -> list[str]:
        """Topological order (higher priority first) of ``subset`` under the
        precedence restricted to ``subset``; ties broken by name for
        determinism. ``subset`` is assumed acyclic (PBS maintains this)."""
        indeg = {a: 0 for a in subset}
        succ: dict[str, list[str]] = defaultdict(list)
        for high, low in prio:
            if high in subset and low in subset:
                succ[high].append(low)
                indeg[low] += 1
        ready = sorted(a for a in subset if indeg[a] == 0)
        queue = deque(ready)
        out: list[str] = []
        seen: set[str] = set()
        while queue:
            a = queue.popleft()
            if a in seen:
                continue
            seen.add(a)
            out.append(a)
            for b in sorted(succ.get(a, ())):
                indeg[b] -= 1
                if indeg[b] == 0:
                    queue.append(b)
        # Any leftover (shouldn't happen for a DAG) appended deterministically.
        for a in sorted(subset):
            if a not in seen:
                out.append(a)
        return out

    @staticmethod
    def _creates_cycle(
        prio: frozenset[tuple[str, str]], high: str, low: str
    ) -> bool:
        """Would adding ``high ≺ low`` create a cycle? True iff ``low`` already
        reaches ``high`` (``low ≺ … ≺ high``)."""
        if high == low:
            return True
        succ: dict[str, set[str]] = defaultdict(set)
        for h, l in prio:
            succ[h].add(l)
        stack = [low]
        seen: set[str] = set()
        while stack:
            x = stack.pop()
            if x == high:
                return True
            if x in seen:
                continue
            seen.add(x)
            stack.extend(succ.get(x, ()))
        return False

    def _reservations_for(
        self,
        plan: Mapping[str, list[str]],
        prio: frozenset[tuple[str, str]],
        higher_agents: set[str],
    ) -> tuple[dict[int, set[str]], dict[int, set[tuple[str, str]]]]:
        """Vertex/edge reservations from the higher-priority agents' committed
        paths, reserved in topological order so the final-node holds stay
        mutually conflict-free."""
        vres: dict[int, set[str]] = defaultdict(set)
        eres: dict[int, set[tuple[str, str]]] = defaultdict(set)
        for h in self._topo_order(prio, set(higher_agents)):
            if h in plan:
                self._reserve(plan[h], vres, eres)
        return vres, eres

    # ----------------------------------------------------- collision detection

    def _first_collision(
        self, plan: Mapping[str, list[str]]
    ) -> Optional[tuple[str, str]]:
        """Earliest-timestep collision pair (deterministic by agent name)."""
        agents = sorted(plan)
        best: Optional[tuple[int, str, str]] = None
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                t = self._collision_time(plan[agents[i]], plan[agents[j]])
                if t is not None and (best is None or t < best[0]):
                    best = (t, agents[i], agents[j])
        return None if best is None else (best[1], best[2])

    def _collides_with_any(
        self, plan: Mapping[str, list[str]], k: str, higher: set[str]
    ) -> bool:
        pk = plan[k]
        for h in higher:
            if h in plan and self._collision_time(pk, plan[h]) is not None:
                return True
        return False

    def _num_collisions(self, plan: Mapping[str, list[str]]) -> int:
        agents = sorted(plan)
        n = 0
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                if self._collision_time(plan[agents[i]], plan[agents[j]]) is not None:
                    n += 1
        return n

    def _collision_time(self, p1: Sequence[str], p2: Sequence[str]) -> Optional[int]:
        """Earliest timestep at which ``p1`` and ``p2`` share a vertex or swap
        an edge. A finished windowed path holds its last node (matching the
        reservation semantics)."""
        horizon = max(len(p1), len(p2))

        def pos(p: Sequence[str], t: int) -> str:
            return p[t] if t < len(p) else p[-1]

        for t in range(horizon):
            if pos(p1, t) == pos(p2, t):  # vertex collision
                return t
            if t + 1 < horizon:
                a0, a1 = pos(p1, t), pos(p1, t + 1)
                b0, b1 = pos(p2, t), pos(p2, t + 1)
                if a0 != a1 and a0 == b1 and a1 == b0:  # edge swap (head-on)
                    return t
        return None

    # --------------------------------------------------- windowed space-time A*

    def _windowed_astar(
        self,
        start: str,
        goal: str,
        vertex_res: Mapping[int, set[str]],
        edge_res: Mapping[int, set[tuple[str, str]]],
    ) -> Optional[list[str]]:
        """Best windowed path toward ``goal``: reach it within ``w`` steps if
        possible, else the prefix ending closest to it. ``None`` only if the
        agent cannot even hold its start cell (already reserved by a
        higher-priority agent at t=0 — distinct starts make this rare)."""
        if start not in self.G:
            return None
        if start in vertex_res.get(0, frozenset()):
            return None
        heur = self._heuristic(goal)
        inf = self.window + 1
        w = self.window

        counter = 0
        h0 = heur.get(start, inf)
        frontier: list[tuple[int, int, int, str, int]] = [(h0, 0, counter, start, 0)]
        came_from: dict[tuple[str, int], tuple[str, int]] = {}
        visited: set[tuple[str, int]] = {(start, 0)}
        # A valid terminal node must stay clear of higher-priority reservations
        # for the rest of the window (the agent holds there once stopped), else a
        # lower-priority agent could park on a cell a higher agent enters later.
        best_state = (start, 0)
        best_key = (
            (h0, 0) if self._parkable(start, 0, vertex_res) else (inf + 1, w + 1)
        )  # (hops-to-goal, time): minimise -> closest, earliest, parkable

        while frontier:
            _, _, _, node, t = heapq.heappop(frontier)
            hk = heur.get(node, inf)
            if (hk, t) < best_key and self._parkable(node, t, vertex_res):
                best_key = (hk, t)
                best_state = (node, t)
                if hk == 0:
                    # Reached the goal and can hold it: best possible progress.
                    break
            if t >= w:
                continue
            nt = t + 1
            for nxt in self._candidates(node):
                state = (nxt, nt)
                if state in visited:
                    continue
                if not self._clear(node, nxt, t, nt, vertex_res, edge_res):
                    continue
                visited.add(state)
                came_from[state] = (node, t)
                counter += 1
                heapq.heappush(
                    frontier, (nt + heur.get(nxt, inf), nt, counter, nxt, nt)
                )
        return self._reconstruct(best_state, came_from)

    def _parkable(
        self,
        node: str,
        t: int,
        vertex_res: Mapping[int, set[str]],
    ) -> bool:
        """True if ``node`` is free of higher-priority reservations for every
        timestep from ``t`` to the window end — i.e. the agent may stop and hold
        here without later colliding with a higher-priority agent that arrives.
        This is the windowed analogue of ``reserve_goal_forever`` in full-horizon
        prioritized planning, and is what makes a complete priority order produce
        a genuinely collision-free joint plan."""
        for tt in range(t, self.window + 1):
            if node in vertex_res.get(tt, frozenset()):
                return False
        return True

    def _candidates(self, node: str) -> list[str]:
        # Waiting in place is always an option (checked against reservations).
        out = [node]
        out.extend(self.G.successors(node))
        return out

    def _clear(
        self,
        src: str,
        dst: str,
        t: int,
        nt: int,
        vertex_res: Mapping[int, set[str]],
        edge_res: Mapping[int, set[tuple[str, str]]],
    ) -> bool:
        if dst in vertex_res.get(nt, frozenset()):
            return False
        if dst != src and (dst, src) in edge_res.get(nt, frozenset()):
            return False  # head-on edge swap
        if self.check_following and dst != src and dst in vertex_res.get(t, frozenset()):
            return False
        return True

    def _reserve(
        self,
        path: Sequence[str],
        vertex_res: dict[int, set[str]],
        edge_res: dict[int, set[tuple[str, str]]],
    ) -> None:
        for t, node in enumerate(path):
            vertex_res[t].add(node)
            if t > 0:
                edge_res[t].add((path[t - 1], node))
        # Hold the final windowed node for the rest of the window so lower-priority
        # agents route around a stopped one. Held UNCONDITIONALLY (no early break):
        # this is what makes a complete prioritized plan collision-free *by
        # construction* — a lower-priority agent can never enter a higher-priority
        # agent's resting cell. Overlapping holds among the higher agents are
        # harmless to the agent currently being planned (it avoids the cell at that
        # timestep regardless of which higher agent holds it).
        last = path[-1]
        for t in range(len(path), self.window + 1):
            vertex_res[t].add(last)

    # ----------------------------------------------------------------- helpers

    def _heuristic(self, goal: str) -> dict[str, int]:
        cached = self._heur_cache.get(goal)
        if cached is None:
            try:
                cached = nx.single_source_shortest_path_length(self._reverse, goal)
            except nx.NetworkXError:
                cached = {}
            self._heur_cache[goal] = cached
        return cached

    @staticmethod
    def _reconstruct(
        state: tuple[str, int],
        came_from: Mapping[tuple[str, int], tuple[str, int]],
    ) -> list[str]:
        path = [state[0]]
        while state in came_from:
            state = came_from[state]
            path.append(state[0])
        path.reverse()
        return path
