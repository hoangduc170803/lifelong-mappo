"""Plumbing tests for the Sprint 4b GATE-1 success-regime eval harness.

These do NOT check learned performance (the actor is untrained) — they verify
the load → rollout → terminal-detect → Wilson summary pipeline runs and that
the pinned thresholds/budgets are wired correctly.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

# gate_eval inserts the vendored on-policy root onto sys.path at import time;
# import it FIRST so the subsequent onpolicy import resolves.
from src.rl.mappo_onpolicy.gate_eval import PINNED, _eval_args, eval_success
from src.rl.mappo_onpolicy.env_adapter import (
    WarehouseEnvConfig,
    WarehouseOnPolicyEnv,
)

from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor


def _make_checkpoint(tmp: Path, num_agents: int) -> Path:
    """Instantiate a fresh GAT actor and save it as a curriculum-style ckpt."""
    args = _eval_args(num_agents)
    cfg = WarehouseEnvConfig(
        num_agents=num_agents,
        episode_horizon=64,
        task_rate=0.1,
        task_distribution="hotspot",
        knn_agents=args.knn_agents,
    )
    wrap = WarehouseOnPolicyEnv(cfg, auto_reset=False)
    actor = R_Actor(args, wrap.observation_space[0], wrap.action_space[0])
    ckpt = tmp / "actor.pt"
    torch.save(actor.state_dict(), str(ckpt))
    wrap.close()
    return ckpt


class TestGateEval(unittest.TestCase):
    def test_pinned_thresholds_present_for_four_hard_dists(self):
        self.assertEqual(
            set(PINNED),
            {"hotspot", "burst_wave", "betweenness_bottleneck", "bipartite_pickup_dropoff"},
        )
        for d, v in PINNED.items():
            self.assertIn("budget", v)
            self.assertTrue(0 < v["threshold"] < 1)

    def test_eval_success_schema_and_bounds(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            ckpt = _make_checkpoint(tmp, num_agents=5)
            summary = eval_success(
                ckpt,
                distribution="hotspot",
                seeds=[0, 1, 2],
                num_agents=5,
                task_budget=1,
                step_cap=48,
            )
        for key in (
            "distribution",
            "success_count",
            "success_rate",
            "success_wilson95",
            "gate_threshold",
            "passes",
            "n",
        ):
            self.assertIn(key, summary)
        self.assertEqual(summary["n"], 3)
        self.assertLessEqual(summary["success_count"], 3)
        self.assertIsInstance(summary["passes"], bool)
        lo, hi = summary["success_wilson95"]
        self.assertLessEqual(lo, hi)
        self.assertEqual(summary["gate_threshold"], PINNED["hotspot"]["threshold"])

    def test_passes_requires_wilson_lower_above_threshold(self):
        # An untrained actor almost surely does not clear a 0.726 bar; check the
        # pass rule is wired (passes == lower > threshold), not just success>0.
        with tempfile.TemporaryDirectory() as d:
            ckpt = _make_checkpoint(Path(d), num_agents=5)
            summary = eval_success(
                ckpt, "hotspot", seeds=[0, 1], num_agents=5, task_budget=1, step_cap=48
            )
        lo = summary["success_wilson95"][0]
        self.assertEqual(summary["passes"], lo > summary["gate_threshold"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
