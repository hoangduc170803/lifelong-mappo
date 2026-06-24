"""Tests for Sprint 2 benchmark visualization summaries."""

from __future__ import annotations

import importlib.util
import math
import unittest

PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None

if PANDAS_AVAILABLE:
    import pandas as pd

    from src.baselines.plot_benchmarks import (
        _asymmetric_yerr,
        _missing_bar_height,
        normalize_benchmark_df,
        summarize_benchmarks,
    )


@unittest.skipUnless(PANDAS_AVAILABLE, "pandas not installed")
class TestPlotBenchmarks(unittest.TestCase):
    def test_summarize_uses_successful_runs_for_performance_metrics(self):
        df = pd.DataFrame(
            [
                {
                    "baseline": "priority_search",
                    "seed": 0,
                    "num_agents": 10,
                    "success": "True",
                    "instance_makespan": "50",
                    "waiting_time_agent_steps": "20",
                    "throughput_tasks_per_1000_steps": "200",
                    "elapsed_s": "0.10",
                },
                {
                    "baseline": "priority_search",
                    "seed": 1,
                    "num_agents": 10,
                    "success": "False",
                    "instance_makespan": "999",
                    "waiting_time_agent_steps": "999",
                    "throughput_tasks_per_1000_steps": "0",
                    "elapsed_s": "0.30",
                },
            ]
        )

        summary = summarize_benchmarks(df)

        self.assertEqual(len(summary), 1)
        row = summary.iloc[0]
        self.assertEqual(row["runs"], 2)
        self.assertAlmostEqual(row["success_rate"], 0.5)
        self.assertAlmostEqual(row["instance_makespan_mean"], 50.0)
        self.assertAlmostEqual(row["waiting_time_agent_steps_mean"], 20.0)
        self.assertAlmostEqual(row["throughput_tasks_per_1000_steps_mean"], 200.0)
        self.assertAlmostEqual(row["elapsed_s_mean"], 0.20)
        self.assertTrue(math.isnan(row["instance_makespan_std"]))

    def test_normalize_rejects_missing_required_columns(self):
        with self.assertRaises(ValueError):
            normalize_benchmark_df(pd.DataFrame([{"baseline": "priority_search"}]))

    def test_asymmetric_yerr_clips_lower_error_at_zero(self):
        yerr = _asymmetric_yerr([4.0, 10.0, math.nan], [6.0, 2.0, math.nan])

        self.assertEqual(yerr[0], [4.0, 2.0, 0.0])
        self.assertEqual(yerr[1], [6.0, 2.0, 0.0])

    def test_missing_bar_height_is_small_positive_placeholder(self):
        self.assertAlmostEqual(_missing_bar_height([math.nan, 100.0]), 1.5)
        self.assertAlmostEqual(_missing_bar_height([math.nan]), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
