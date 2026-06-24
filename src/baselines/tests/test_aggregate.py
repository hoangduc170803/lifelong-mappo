"""Tests for benchmark CSV aggregation."""

from __future__ import annotations

import math
import unittest

from src.baselines.aggregate import aggregate_rows, render_markdown


class TestAggregate(unittest.TestCase):
    def test_groups_rows_and_counts_failure_reasons(self):
        summary = aggregate_rows(
            [
                {
                    "baseline": "priority_search",
                    "num_agents": "10",
                    "success": "True",
                    "instance_makespan": "50",
                    "makespan_over_lower_bound": "1.25",
                    "waiting_time_agent_steps": "20",
                    "throughput_tasks_per_1000_steps": "200",
                    "elapsed_s": "0.1",
                    "failure_reason": "",
                },
                {
                    "baseline": "priority_search",
                    "num_agents": "10",
                    "success": "False",
                    "instance_makespan": "0",
                    "makespan_over_lower_bound": "inf",
                    "waiting_time_agent_steps": "0",
                    "throughput_tasks_per_1000_steps": "0",
                    "elapsed_s": "0.2",
                    "failure_reason": "timeout",
                },
            ],
            ci_method="bootstrap",
            bootstrap_samples=50,
            seed=123,
        )
        self.assertEqual(len(summary), 1)
        row = summary[0]
        self.assertEqual(row["runs"], 2)
        self.assertAlmostEqual(row["success_rate"], 0.5)
        self.assertGreaterEqual(row["success_rate_ci_low"], 0.0)
        self.assertLessEqual(row["success_rate_ci_high"], 1.0)
        self.assertAlmostEqual(row["mean_makespan_success"], 50.0)
        self.assertTrue(math.isnan(row["std_makespan_success"]))
        self.assertAlmostEqual(row["mean_makespan_all_penalized"], 281.0)
        self.assertAlmostEqual(row["mean_wait_all_penalized"], 2570.0)
        self.assertAlmostEqual(row["mean_throughput_all"], 100.0)
        self.assertEqual(row["failure_reasons"], "timeout:1")
        rendered = render_markdown(summary)
        self.assertIn("priority_search", rendered)
        self.assertIn("mean_makespan_success", rendered)
        self.assertIn("mean_makespan_all_penalized", rendered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
