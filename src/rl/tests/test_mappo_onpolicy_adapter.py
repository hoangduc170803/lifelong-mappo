"""Smoke + shape tests for the on-policy MAPPO adapter stack."""

from __future__ import annotations

import argparse
import os
import sys
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import torch

# Ensure vendored on-policy is importable for the whole test module.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ON_POLICY = _REPO_ROOT / "on-policy"
if str(_ON_POLICY) not in sys.path:
    sys.path.insert(0, str(_ON_POLICY))

from gymnasium import spaces  # noqa: E402

from src.env.compass_mapper import NUM_ACTIONS  # noqa: E402
from src.env.warehouse_env import WarehouseEnv  # noqa: E402
from src.rl.mappo_onpolicy.env_adapter import (  # noqa: E402
    WarehouseEnvConfig,
    WarehouseOnPolicyEnv,
)
from src.rl.mappo_onpolicy.vec_wrapper import DummyVecEnv  # noqa: E402
from src.routing.astar import AStarRouter  # noqa: E402


def _grid_graph(width: int = 3, height: int = 3) -> nx.DiGraph:
    graph = nx.DiGraph()
    for y in range(height):
        for x in range(width):
            graph.add_node(f"{x},{y}", x=float(x), y=float(y))
    for y in range(height):
        for x in range(width):
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxv, nyv = x + dx, y + dy
                if 0 <= nxv < width and 0 <= nyv < height:
                    graph.add_edge(f"{x},{y}", f"{nxv},{nyv}", length=1.0)
    return graph


def _tiny_env(num_agents: int = 3, seed: int = 0) -> WarehouseEnv:
    graph = _grid_graph()
    router = AStarRouter(graph, precompute=True)
    return WarehouseEnv(
        graph=graph,
        router=router,
        num_agents=num_agents,
        episode_horizon=8,
        task_rate=0.1,
        knn_agents=2,
        seed=seed,
    )


def _tiny_adapter(num_agents: int = 3, seed: int = 0) -> WarehouseOnPolicyEnv:
    return WarehouseOnPolicyEnv(env=_tiny_env(num_agents=num_agents, seed=seed))


class TestEnvAdapter(unittest.TestCase):
    def test_reset_returns_per_agent_arrays_with_enriched_share_obs(self):
        adapter = _tiny_adapter(num_agents=3)
        obs, share_obs, avail = adapter.reset(seed=0)

        self.assertEqual(obs.shape, (3, adapter.obs_dim))
        self.assertEqual(share_obs.shape, (3, adapter.share_obs_dim))
        self.assertGreater(adapter.share_obs_dim, adapter.obs_dim * 3)
        self.assertEqual(avail.shape, (3, NUM_ACTIONS))
        self.assertTrue(np.all(share_obs[0] == share_obs[1]))
        self.assertTrue(np.all(share_obs[1] == share_obs[2]))
        self.assertEqual(obs.dtype, np.float32)
        self.assertEqual(share_obs.dtype, np.float32)
        self.assertEqual(avail.dtype, np.float32)
        for space in adapter.observation_space:
            self.assertIsInstance(space, spaces.Box)
            self.assertEqual(space.shape, (adapter.obs_dim,))
        for space in adapter.share_observation_space:
            self.assertIsInstance(space, spaces.Box)
            self.assertEqual(space.shape, (adapter.share_obs_dim,))
        for space in adapter.action_space:
            self.assertIsInstance(space, spaces.Discrete)
            self.assertEqual(space.n, NUM_ACTIONS)

        global_start = adapter.obs_dim * adapter.num_agents
        expected_pending = adapter.env._task_gen.num_pending() / adapter.env.max_tasks_in_pool
        self.assertAlmostEqual(float(share_obs[0, global_start + 1]), expected_pending)

    def test_share_obs_dim_matches_spec_formula(self):
        """D3 acceptance: critic input dimension matches the documented spec.

        share_obs = concat(local obs) + global state, where the global state is
        [step, pending, in_flight, 5*K task descriptors, N idle flags, congestion].
        """
        num_agents = 3
        adapter = _tiny_adapter(num_agents=num_agents)
        k = adapter.pending_task_features_k
        self.assertEqual(k, 4)  # default K from spec (nearest-4 pending tasks)

        expected_global = 3 + 5 * k + num_agents + 1
        expected_share = adapter.obs_dim * num_agents + expected_global
        self.assertEqual(adapter.global_state_dim, expected_global)
        self.assertEqual(adapter.share_obs_dim, expected_share)
        self.assertEqual(
            adapter.env.global_state_dim(k), expected_global
        )

    def test_global_state_layout_features(self):
        """Cover D3 features (iii) idle flags and (iv) congestion proxy slots."""
        num_agents = 3
        adapter = _tiny_adapter(num_agents=num_agents)
        obs, share_obs, _ = adapter.reset(seed=0)
        k = adapter.pending_task_features_k
        global_start = adapter.obs_dim * num_agents
        glob = share_obs[0, global_start:]

        self.assertEqual(glob.shape[0], adapter.global_state_dim)
        # step-progress, pending, in-flight pressures are all finite & >= 0.
        self.assertTrue(np.all(np.isfinite(glob)))
        self.assertGreaterEqual(float(glob[0]), 0.0)
        # Idle-flag block is strictly boolean-valued.
        idle_start = 3 + 5 * k
        idle_block = glob[idle_start:idle_start + num_agents]
        self.assertTrue(np.all((idle_block == 0.0) | (idle_block == 1.0)))
        # Congestion proxy is the final slot; the EMA starts at 0 on reset.
        self.assertEqual(float(glob[-1]), 0.0)

    def test_step_shapes_and_done_flag(self):
        adapter = _tiny_adapter(num_agents=2)
        adapter.reset(seed=0)
        actions = np.full((2, 1), NUM_ACTIONS - 1, dtype=np.float32)  # WAIT
        obs, share, rewards, dones, info, avail = adapter.step(actions)

        self.assertEqual(obs.shape, (2, adapter.obs_dim))
        self.assertEqual(rewards.shape, (2, 1))
        self.assertEqual(dones.shape, (2,))
        self.assertEqual(dones.dtype, bool)
        self.assertEqual(avail.shape, (2, NUM_ACTIONS))
        self.assertIn("tasks_completed_total", info)
        self.assertIn("bad_transition", info)

    def test_step_level_info_mismatch_is_not_silent(self):
        adapter = _tiny_adapter(num_agents=2)

        with self.assertRaises(AssertionError):
            adapter._merge_infos(
                {
                    "agent_0": {"tasks_pending": 1},
                    "agent_1": {"tasks_pending": 2},
                }
            )


class TestActionMaskGate(unittest.TestCase):
    def test_policy_never_samples_masked_actions(self):
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import (
            R_MAPPOPolicy,
        )

        args = argparse.Namespace(
            hidden_size=16,
            layer_N=1,
            use_orthogonal=True,
            gain=0.01,
            use_feature_normalization=True,
            use_ReLU=True,
            use_popart=False,
            use_valuenorm=True,
            use_naive_recurrent_policy=False,
            use_recurrent_policy=False,
            recurrent_N=1,
            data_chunk_length=10,
            lr=5e-4,
            critic_lr=5e-4,
            opti_eps=1e-5,
            weight_decay=0.0,
            algorithm_name="mappo",
            use_policy_active_masks=True,
            use_max_grad_norm=True,
            max_grad_norm=10.0,
            use_clipped_value_loss=True,
            use_huber_loss=True,
            huber_delta=10.0,
            use_value_active_masks=True,
            use_gae=True,
            use_proper_time_limits=False,
            gamma=0.99,
            gae_lambda=0.95,
            ppo_epoch=4,
            num_mini_batch=1,
            value_loss_coef=1.0,
            entropy_coef=0.01,
            clip_param=0.2,
            stacked_frames=1,
            use_stacked_frames=False,
        )
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        share_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8 * 4,), dtype=np.float32
        )
        act_space = spaces.Discrete(NUM_ACTIONS)
        policy = R_MAPPOPolicy(args, obs_space, share_space, act_space)

        # All slots masked except WAIT (slot 8): every sampled action must be 8.
        batch = 16
        obs = np.random.randn(batch, 8).astype(np.float32)
        share = np.random.randn(batch, 8 * 4).astype(np.float32)
        rnn = np.zeros((batch, 1, 16), dtype=np.float32)
        masks = np.ones((batch, 1), dtype=np.float32)
        avail = np.zeros((batch, NUM_ACTIONS), dtype=np.float32)
        avail[:, NUM_ACTIONS - 1] = 1.0

        with torch.no_grad():
            _, actions, _, _, _ = policy.get_actions(share, obs, rnn, rnn, masks, avail)
        actions_np = actions.cpu().numpy().reshape(-1)
        self.assertTrue(np.all(actions_np == NUM_ACTIONS - 1))


class TestDummyVecEnv(unittest.TestCase):
    def test_vec_reset_step_have_leading_thread_dim(self):
        def make(rank: int) -> WarehouseOnPolicyEnv:
            return WarehouseOnPolicyEnv(env=_tiny_env(num_agents=2, seed=rank))

        vec = DummyVecEnv([make, make])
        obs, share, avail = vec.reset(seeds=[0, 1])
        self.assertEqual(obs.shape, (2, 2, vec.envs[0].obs_dim))
        self.assertEqual(share.shape, (2, 2, vec.envs[0].share_obs_dim))
        self.assertEqual(avail.shape, (2, 2, NUM_ACTIONS))

        actions = np.full((2, 2, 1), NUM_ACTIONS - 1, dtype=np.float32)
        obs2, share2, rewards, dones, infos, avail2 = vec.step(actions)
        self.assertEqual(obs2.shape, (2, 2, vec.envs[0].obs_dim))
        self.assertEqual(rewards.shape, (2, 2, 1))
        self.assertEqual(dones.shape, (2, 2))
        self.assertEqual(avail2.shape, (2, 2, NUM_ACTIONS))
        self.assertEqual(len(infos), 2)
        vec.close()


class TestWarehouseRunnerSmoke(unittest.TestCase):
    def test_smoke_train_runs_one_update(self):
        from src.rl.mappo_onpolicy.config import get_warehouse_config
        from src.rl.mappo_onpolicy.train import main

        run_dir = _REPO_ROOT / "results" / "sprint3" / "onpolicy_smoke_test"
        argv = [
            "--num_agents_target",
            "2",
            "--episode_length",
            "8",
            "--num_env_steps",
            "16",
            "--hidden_size",
            "16",
            "--layer_N",
            "1",
            "--experiment_name",
            "unittest",
            "--seed",
            "0",
            "--save_interval",
            "100",
            "--log_interval",
            "1",
            "--results_dir",
            str(run_dir),
        ]
        rc = main(argv)
        self.assertEqual(rc, 0)
        # Ensure the runner produced a checkpoint and tensorboard log dir.
        out = run_dir / "unittest_seed0"
        self.assertTrue((out / "logs").exists())
        self.assertTrue((out / "models" / "actor.pt").exists())
        self.assertTrue((out / "models" / "critic.pt").exists())


if __name__ == "__main__":
    unittest.main()
