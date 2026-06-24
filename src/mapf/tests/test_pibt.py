"""Tests for the PIBT one-step planner (Okumura 2022)."""

from __future__ import annotations

import unittest

import networkx as nx

from src.mapf.pibt import PIBT


def _line(n=5):
    """Bidirectional line 0-1-...-(n-1) with unit lengths."""
    G = nx.DiGraph()
    for x in range(n):
        G.add_node(str(x), x=float(x), y=0.0)
    for x in range(n - 1):
        G.add_edge(str(x), str(x + 1), weight=1.0)
        G.add_edge(str(x + 1), str(x), weight=1.0)
    return G


def _grid(w=4, h=4):
    G = nx.DiGraph()
    for x in range(w):
        for y in range(h):
            G.add_node(f"{x},{y}", x=float(x), y=float(y))
    for x in range(w):
        for y in range(h):
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx_, ny_ = x + dx, y + dy
                if 0 <= nx_ < w and 0 <= ny_ < h:
                    G.add_edge(f"{x},{y}", f"{nx_},{ny_}", weight=1.0)
    return G


def _dist_fn(G):
    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))

    def dist(u, v):
        return float(lengths[u].get(v, float("inf")))

    return dist


def _assert_valid_joint_move(testcase, positions, moves):
    """No vertex conflicts, no direct swaps, moves only along edges/stays."""
    targets = list(moves.values())
    testcase.assertEqual(len(targets), len(set(targets)), "vertex conflict")
    for a, v in moves.items():
        for b, w in moves.items():
            if a < b:
                swapped = v == positions[b] and w == positions[a] and v != w
                testcase.assertFalse(swapped, f"swap between {a} and {b}")


class TestPIBT(unittest.TestCase):
    def test_single_agent_moves_toward_goal(self):
        G = _line(5)
        pibt = PIBT(G, _dist_fn(G), seed=0)
        moves = pibt.step({"a": "0"}, {"a": "4"}, {"a": 1.0})
        self.assertEqual(moves["a"], "1")

    def test_agent_at_goal_stays(self):
        G = _line(5)
        pibt = PIBT(G, _dist_fn(G), seed=0)
        moves = pibt.step({"a": "2"}, {"a": "2"}, {"a": 1.0})
        self.assertEqual(moves["a"], "2")

    def test_high_priority_pushes_idle_occupant(self):
        """Priority inheritance: the blocker is forced to vacate."""
        G = _line(5)
        pibt = PIBT(G, _dist_fn(G), seed=0)
        positions = {"hi": "1", "lo": "2"}
        goals = {"hi": "4", "lo": "2"}  # lo idles exactly on hi's path
        moves = pibt.step(positions, goals, {"hi": 10.0, "lo": 0.0})
        _assert_valid_joint_move(self, positions, moves)
        self.assertEqual(moves["hi"], "2")
        self.assertNotEqual(moves["lo"], "2")  # pushed off its node

    def test_head_on_no_swap(self):
        """Two agents facing each other must never swap through an edge."""
        G = _line(5)
        pibt = PIBT(G, _dist_fn(G), seed=0)
        positions = {"a": "1", "b": "2"}
        goals = {"a": "4", "b": "0"}
        moves = pibt.step(positions, goals, {"a": 5.0, "b": 1.0})
        _assert_valid_joint_move(self, positions, moves)

    def test_boxed_agent_stays(self):
        """Nowhere to go (allowed filter blocks everything) -> stay."""
        G = _line(3)
        pibt = PIBT(G, _dist_fn(G), seed=0)
        moves = pibt.step(
            {"a": "1"},
            {"a": "2"},
            {"a": 1.0},
            allowed={"a": set()},  # mask forbids every neighbor
        )
        self.assertEqual(moves["a"], "1")

    def test_allowed_filter_restricts_candidates(self):
        G = _line(5)
        pibt = PIBT(G, _dist_fn(G), seed=0)
        # Goal is to the right but the mask only allows moving left.
        moves = pibt.step(
            {"a": "2"},
            {"a": "4"},
            {"a": 1.0},
            allowed={"a": {"1"}},
        )
        self.assertIn(moves["a"], {"1", "2"})

    def test_corridor_chain_makes_progress(self):
        """A convoy in a corridor advances (following is allowed in PIBT)."""
        G = _line(6)
        pibt = PIBT(G, _dist_fn(G), seed=0)
        positions = {"a": "0", "b": "1", "c": "2"}
        goals = {"a": "5", "b": "5", "c": "5"}
        pr = {"a": 1.0, "b": 2.0, "c": 3.0}
        moves = pibt.step(positions, goals, pr)
        _assert_valid_joint_move(self, positions, moves)
        self.assertEqual(moves["c"], "3")
        self.assertEqual(moves["b"], "2")
        self.assertEqual(moves["a"], "1")

    def test_grid_many_agents_valid_and_deterministic(self):
        G = _grid(4, 4)
        dist = _dist_fn(G)
        positions = {
            "a": "0,0", "b": "3,3", "c": "0,3", "d": "3,0", "e": "1,1",
        }
        goals = {
            "a": "3,3", "b": "0,0", "c": "3,0", "d": "0,3", "e": "2,2",
        }
        pr = {"a": 5, "b": 4, "c": 3, "d": 2, "e": 1}
        m1 = PIBT(G, dist, seed=7).step(positions, goals, pr)
        m2 = PIBT(G, dist, seed=7).step(positions, goals, pr)
        _assert_valid_joint_move(self, positions, m1)
        self.assertEqual(m1, m2)

    def test_multi_step_rollout_reaches_goals(self):
        """Iterated PIBT on a grid: everyone reaches their goal eventually."""
        G = _grid(4, 4)
        dist = _dist_fn(G)
        pibt = PIBT(G, dist, seed=3)
        positions = {"a": "0,0", "b": "3,3", "c": "0,3"}
        goals = {"a": "3,3", "b": "0,0", "c": "3,0"}
        priorities = {a: 0.0 for a in positions}
        for _ in range(40):
            moves = pibt.step(positions, goals, priorities)
            _assert_valid_joint_move(self, positions, moves)
            positions = dict(moves)
            for a in positions:
                if positions[a] == goals[a]:
                    priorities[a] = 0.0
                else:
                    priorities[a] += 1.0
            if all(positions[a] == goals[a] for a in positions):
                break
        self.assertTrue(all(positions[a] == goals[a] for a in positions))


if __name__ == "__main__":
    unittest.main(verbosity=2)
