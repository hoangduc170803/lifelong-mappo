"""Tests for Sprint 3.5 gate evidence runner."""

from __future__ import annotations

import importlib.util
import unittest

from src.baselines.sprint35_gate import (
    ROLLING_HORIZON_BASELINE,
    RollingHorizonPrioritySearchPlanner,
    build_cbs_grid,
    run_cbs_reference,
    run_pp_optimality_probe,
    run_rolling_horizon_spot_check,
    sample_grid_instance,
    sample_warehouse_stress_instance,
)
from src.map_parser import parse_opentcs_map
from src.baselines.benchmark import DEFAULT_MAP_FILE


class TestSprint35GateRunner(unittest.TestCase):
    def test_sample_grid_instance_is_distinct(self):
        G = build_cbs_grid(size=5)
        starts, goals = sample_grid_instance(G, num_agents=3, seed=7)

        self.assertEqual(len(starts), 3)
        self.assertEqual(len(goals), 3)
        self.assertEqual(len(set(starts.values()) | set(goals.values())), 6)

    def test_hotspot_and_burst_stress_instances_are_distinct(self):
        G = parse_opentcs_map(str(DEFAULT_MAP_FILE), restrict_to_largest_scc=True)
        for distribution in (
            "hotspot",
            "burst_wave",
            "betweenness_bottleneck",
            "bipartite_pickup_dropoff",
        ):
            starts, goals = sample_warehouse_stress_instance(
                G,
                num_agents=5,
                seed=3,
                distribution=distribution,
            )
            self.assertEqual(len(starts), 5, f"{distribution}: starts")
            self.assertEqual(len(goals), 5, f"{distribution}: goals")
            self.assertEqual(len(set(starts.values())), 5, f"{distribution}: starts unique")
            self.assertEqual(len(set(goals.values())), 5, f"{distribution}: goals unique")

    def test_bipartite_distribution_uses_disjoint_endpoint_pools(self):
        from src.baselines.sprint35_gate import _outer_to_center_pools

        G = parse_opentcs_map(str(DEFAULT_MAP_FILE), restrict_to_largest_scc=True)
        start_pool, goal_pool = _outer_to_center_pools(G, min_size=40)
        starts, goals = sample_warehouse_stress_instance(
            G,
            num_agents=5,
            seed=19,
            distribution="bipartite_pickup_dropoff",
        )
        self.assertTrue(set(start_pool).isdisjoint(set(goal_pool)))
        for start in starts.values():
            self.assertIn(start, set(start_pool))
        for goal in goals.values():
            self.assertIn(goal, set(goal_pool))

    def test_betweenness_distribution_goals_lie_in_topology_pool(self):
        """Issue #4 fix: goal pool must come from betweenness centrality
        ranking, not from Euclidean centroid distance."""
        from src.baselines.sprint35_gate import _betweenness_bottleneck_pool

        G = parse_opentcs_map(str(DEFAULT_MAP_FILE), restrict_to_largest_scc=True)
        expected_pool = set(_betweenness_bottleneck_pool(G, min_size=24))
        _, goals = sample_warehouse_stress_instance(
            G,
            num_agents=5,
            seed=11,
            distribution="betweenness_bottleneck",
        )
        for goal in goals.values():
            self.assertIn(goal, expected_pool)

    def test_pp_optimality_probe_brackets_lower_bound(self):
        """Issue #1 fix: exhaustive PP enumeration must report a min-PP
        makespan >= MAPF-IS lower bound for every successful instance,
        and report at least one success when N is small (3 agents).

        ``max_time`` defaults to 256 because warehouse paths regularly exceed
        100 steps even for sparse 3-agent uniform instances; smaller horizons
        cause spurious zero-success rows."""
        rows = run_pp_optimality_probe(
            seeds=[0, 1],
            agent_counts=[3],
            max_time=256,
        )
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row.permutations_attempted, 6)  # 3!
            self.assertGreaterEqual(row.permutations_succeeded, 1)
            self.assertGreaterEqual(row.min_pp_makespan, row.lower_bound_steps)
            self.assertGreaterEqual(row.min_pp_over_lb, 1.0)

    def test_rolling_horizon_spot_check_writes_benchmark_shape(self):
        rows = run_rolling_horizon_spot_check(
            seeds=[0],
            agent_counts=[3],
            distributions=["hotspot"],
            max_time=64,
            replan_horizon=8,
            jobs=1,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["baseline"], ROLLING_HORIZON_BASELINE)
        self.assertEqual(rows[0]["distribution"], "hotspot")
        self.assertIn("diagnostics_json", rows[0])

    def test_rolling_horizon_planner_solves_simple_swap_with_replanning(self):
        G = build_cbs_grid(size=4)
        planner = RollingHorizonPrioritySearchPlanner(
            G,
            max_time=16,
            replan_horizon=2,
            max_orders=4,
        )
        result = planner.plan(
            starts={"agv_0": "0,0", "agv_1": "3,3"},
            goals={"agv_0": "3,3", "agv_1": "0,0"},
            seed=0,
        )
        self.assertTrue(result.success, result.diagnostics)
        self.assertEqual(result.solver, ROLLING_HORIZON_BASELINE)

    @unittest.skipUnless(
        importlib.util.find_spec("cbs_mapf"),
        "cbs-mapf not installed",
    )
    def test_cbs_reference_uses_external_backend_without_fallback(self):
        rows = run_cbs_reference(
            seeds=[0],
            agent_counts=[1],
            warehouse_probe=False,
            max_time=16,
            cbs_max_iter=20,
            cbs_low_level_max_iter=50,
        )

        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0].cbs_success, rows[0].cbs_diagnostics_json)
        self.assertEqual(rows[0].cbs_solver, "cbs_mapf")
        self.assertEqual(rows[0].pp32_over_cbs, 1.0)

    @unittest.skipUnless(
        importlib.util.find_spec("cbs_mapf"),
        "cbs-mapf not installed",
    )
    def test_cbs_reference_can_run_outer_jobs_in_parallel(self):
        rows = run_cbs_reference(
            seeds=[0, 1],
            agent_counts=[1],
            warehouse_probe=False,
            cbs_jobs=2,
            max_time=16,
            cbs_max_iter=20,
            cbs_low_level_max_iter=50,
        )

        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row.cbs_success for row in rows))
        self.assertEqual([row.seed for row in rows], [0, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
