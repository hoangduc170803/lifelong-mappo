"""Tests for episode metrics aggregation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.env.compass_mapper import WAIT_SLOT
from src.utils.episode_logger import EpisodeLogger, append_metrics_csv, write_metrics_json


class TestEpisodeLogger(unittest.TestCase):
    def test_aggregates_env_step_infos(self):
        logger = EpisodeLogger(num_agents=2)
        logger.record_step(
            actions={"a0": WAIT_SLOT, "a1": 0},
            rewards={"a0": -0.1, "a1": 1.0},
            infos={
                "a0": {
                    "validator_interventions": 1,
                    "conflicts_vertex": 1,
                    "conflicts_edge_swap": 0,
                    "conflicts_following": 0,
                    "tasks_completed_total": 0,
                    "tasks_pending": 2,
                    "tasks_in_flight": 1,
                }
            },
        )
        logger.record_step(
            actions={"a0": 1, "a1": WAIT_SLOT},
            rewards={"a0": 1.0, "a1": -0.1},
            infos={
                "a0": {
                    "validator_interventions": 0,
                    "conflicts_vertex": 0,
                    "conflicts_edge_swap": 1,
                    "conflicts_following": 1,
                    "tasks_completed_total": 1,
                    "tasks_pending": 1,
                    "tasks_in_flight": 1,
                }
            },
        )
        metrics = logger.finalize(extra={"policy": "test"})
        self.assertEqual(metrics.steps, 2)
        self.assertEqual(metrics.last_completion_step, 2)
        self.assertEqual(metrics.instance_makespan, 0)
        self.assertEqual(metrics.tasks_completed, 1)
        self.assertEqual(metrics.tasks_generated, 3)
        self.assertEqual(metrics.waiting_time_agent_steps, 2)
        self.assertEqual(metrics.validator_interventions, 1)
        self.assertEqual(metrics.conflicts_total, 3)
        self.assertAlmostEqual(metrics.throughput_tasks_per_1000_steps, 500.0)
        self.assertEqual(metrics.extra["policy"], "test")

    def test_writes_json_and_csv(self):
        logger = EpisodeLogger(num_agents=1)
        metrics = logger.finalize()
        with tempfile.TemporaryDirectory() as tmp:
            json_path = write_metrics_json(metrics, Path(tmp) / "metrics.json")
            csv_path = append_metrics_csv(metrics, Path(tmp) / "metrics.csv")
            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertGreater(json_path.stat().st_size, 0)
            self.assertGreater(csv_path.stat().st_size, 0)

    def test_records_one_shot_mapf_instance_makespan_separately(self):
        logger = EpisodeLogger(num_agents=1)
        metrics = logger.finalize(instance_makespan=7)
        self.assertEqual(metrics.last_completion_step, 0)
        self.assertEqual(metrics.instance_makespan, 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
