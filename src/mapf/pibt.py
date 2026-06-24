"""PIBT — Priority Inheritance with Backtracking (Okumura et al. 2022).

Per-timestep one-shot coordination: given current positions, goals, and
priorities, return the next vertex for every agent such that the joint move
has no vertex conflicts and no direct swaps. Agents greedily move toward
their goals (candidates sorted by distance-to-goal); when a desired vertex is
occupied, the occupant inherits the mover's priority and plans first
(priority inheritance); if the occupant cannot vacate, the mover tries its
next candidate (backtracking).

This module is deliberately decoupled from `WarehouseEnv`:

- the distance oracle is injected (`dist(node, goal) -> float`), so any
  precomputed router works;
- an optional per-agent `allowed` vertex filter lets a wrapper restrict
  candidates to action-mask-valid moves (staying put is always allowed), so
  classical baselines can be compared against MAPPO under the SAME env rules
  (see `src/baselines/pp32_lifelong.py` for why this matters);
- dynamic priorities are the caller's responsibility. The standard lifelong
  scheme (paper §V): priority = steps elapsed since the agent last reached a
  goal, tie-broken by agent id.

Reference: Okumura, Machida, Défago, Tamura, "Priority Inheritance with
Backtracking for Iterative Multi-agent Path Finding", AIJ 2022.
"""

from __future__ import annotations

import random
from typing import Callable, Mapping, Optional

import networkx as nx


class PIBT:
    """One-step PIBT planner over a (directed) graph."""

    def __init__(
        self,
        G: nx.DiGraph,
        dist: Callable[[str, str], float],
        seed: int = 0,
    ):
        self.G = G
        self.dist = dist
        self.rng = random.Random(seed)

    def step(
        self,
        positions: Mapping[str, str],
        goals: Mapping[str, str],
        priorities: Mapping[str, float],
        allowed: Optional[Mapping[str, set[str]]] = None,
        forced: Optional[Mapping[str, str]] = None,
    ) -> Optional[dict[str, str]]:
        """Plan one joint move. Returns {agent: next_node}.

        Guarantees: no two agents share a next node (vertex conflict) and no
        pair swaps positions (edge conflict). Following moves (entering a
        node its occupant is simultaneously vacating) ARE allowed, as in the
        paper; restrict via `allowed` if the env forbids them.

        `forced` pre-assigns specific agents to specific vertices (used by
        LaCAM's low-level constraint tree). Returns ``None`` when the forced
        assignment is infeasible (off-graph move, duplicate claim, or the
        resulting joint move would contain a swap).
        """
        self._positions = dict(positions)
        self._goals = goals
        self._allowed = allowed
        self._occupant = {node: agent for agent, node in positions.items()}
        self._next: dict[str, str] = {}
        self._claimed: dict[str, str] = {}  # node -> claiming agent

        if forced:
            for agent, vertex in forced.items():
                pos = self._positions[agent]
                legal = vertex == pos or vertex in self.G.successors(pos)
                if not legal or vertex in self._claimed:
                    return None
                self._next[agent] = vertex
                self._claimed[vertex] = agent

        order = sorted(
            positions, key=lambda a: (-priorities.get(a, 0.0), a)
        )
        for agent in order:
            if agent not in self._next:
                self._pibt(agent, pusher=None)
        # Belt-and-braces: anyone still unplanned stays put. The env's safety
        # validator remains the final authority on the executed joint move.
        for agent, pos in self._positions.items():
            self._next.setdefault(agent, pos)

        if forced and self._has_swap(self._next):
            return None
        return dict(self._next)

    def _has_swap(self, moves: Mapping[str, str]) -> bool:
        """True if any pair traverses the same edge in opposite directions.

        Unforced PIBT can never produce a swap (the pusher check forbids it),
        but a `forced` assignment bypasses that check, so LaCAM successors
        need this post-validation.
        """
        agents = list(moves)
        for i, a in enumerate(agents):
            pa = self._positions[a]
            for b in agents[i + 1:]:
                pb = self._positions[b]
                if pa != pb and moves[a] == pb and moves[b] == pa:
                    return True
        return False

    # ------------------------------------------------------------------ core

    def _candidates(self, agent: str) -> list[str]:
        pos = self._positions[agent]
        cands = list(self.G.successors(pos)) + [pos]
        if self._allowed is not None:
            allow = self._allowed.get(agent)
            if allow is not None:
                cands = [v for v in cands if v in allow or v == pos]
        goal = self._goals.get(agent, pos)
        self.rng.shuffle(cands)  # random tie-break before stable sort
        cands.sort(key=lambda v: self.dist(v, goal))
        return cands

    def _pibt(self, agent: str, pusher: Optional[str]) -> bool:
        for vertex in self._candidates(agent):
            if vertex in self._claimed:
                continue  # vertex conflict with an already-planned agent
            if pusher is not None and vertex == self._positions[pusher]:
                continue  # direct swap with the agent pushing us
            self._next[agent] = vertex
            self._claimed[vertex] = agent

            occupant = self._occupant.get(vertex)
            if (
                occupant is not None
                and occupant != agent
                and occupant not in self._next
            ):
                # Occupant inherits our priority: it must plan (vacate) now.
                if not self._pibt(occupant, pusher=agent):
                    del self._next[agent]
                    del self._claimed[vertex]
                    continue
            return True

        # No candidate worked: stay put for this step (paper: invalid).
        pos = self._positions[agent]
        if pos not in self._claimed:
            self._next[agent] = pos
            self._claimed[pos] = agent
        # else: our node is claimed by the pusher that targeted it. Recording
        # a stay would create a vertex conflict, so leave ourselves unplanned:
        # returning False forces the pusher to backtrack (undoing that claim),
        # and the top-level loop replans us afterwards (a pushed agent always
        # has lower priority, so its turn comes after the pusher's).
        return False
