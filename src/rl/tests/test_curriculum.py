"""Tests for the Sprint 4b curriculum trainer scaffolding (no training)."""

from __future__ import annotations

import unittest
from pathlib import Path

import tempfile

from src.rl.mappo_onpolicy.curriculum import (
    HARD_DISTRIBUTIONS,
    Stage,
    _actor_path,
    _build_argv,
    _plan_branch_jobs,
    branch_stages,
    stage_status,
    trunk_stages,
)


class TestCurriculumPlan(unittest.TestCase):
    def test_trunk_is_uniform_increasing_agents(self):
        trunk = trunk_stages(smoke=False)
        self.assertEqual([s.num_agents for s in trunk], [5, 10, 15])
        self.assertTrue(all(s.distribution == "uniform" for s in trunk))

    def test_branches_cover_all_hard_distributions_at_n20(self):
        branches = branch_stages(smoke=False)
        self.assertEqual({s.distribution for s in branches}, set(HARD_DISTRIBUTIONS))
        self.assertTrue(all(s.num_agents == 20 for s in branches))

    def test_smoke_stages_are_tiny(self):
        for s in trunk_stages(smoke=True) + branch_stages(smoke=True):
            self.assertLessEqual(s.num_env_steps, 1000)

    def test_argv_sets_gat_encoder_and_warm_start(self):
        stage = branch_stages(smoke=True)[0]
        argv = _build_argv(
            stage,
            seed=2,
            results_dir=Path("results/x"),
            throughput_bonus=0.1,
            threads_per_run=2,
            actor_init_from=Path("prev/actor.pt"),
        )
        self.assertIn("--encoder", argv)
        self.assertEqual(argv[argv.index("--encoder") + 1], "gat")
        self.assertEqual(argv[argv.index("--num_agents_target") + 1], "20")
        self.assertEqual(
            argv[argv.index("--reward_throughput_bonus_per_task") + 1], "0.1"
        )
        self.assertIn("--actor_init_from", argv)

    def test_argv_omits_warm_start_when_none(self):
        stage = trunk_stages(smoke=True)[0]
        argv = _build_argv(
            stage,
            seed=0,
            results_dir=Path("results/x"),
            throughput_bonus=0.1,
            threads_per_run=2,
            actor_init_from=None,
        )
        self.assertNotIn("--actor_init_from", argv)

    def test_actor_path_is_deterministic(self):
        p = _actor_path(Path("results/c"), seed=1, stage_name="n15_uniform")
        self.assertEqual(p.name, "actor.pt")
        self.assertIn("seed1", str(p))
        self.assertIn("n15_uniform", str(p))

    def test_argv_resume_uses_model_dir_and_remaining_steps(self):
        stage = branch_stages(smoke=True)[0]
        argv = _build_argv(
            stage, seed=0, results_dir=Path("results/x"),
            throughput_bonus=0.1, threads_per_run=2, actor_init_from=None,
            resume_model_dir=Path("prev/models"), steps_override=123,
        )
        self.assertIn("--model_dir", argv)
        self.assertNotIn("--actor_init_from", argv)
        self.assertEqual(argv[argv.index("--num_env_steps") + 1], "123")

    def test_stage_status_classifies_done_partial_fresh(self):
        stage = Stage("n20_hotspot", 20, "hotspot", 2_000_000, 1024)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # fresh: no log -> remaining is the full budget
            st, rem = stage_status(root, 0, stage)
            self.assertEqual(st, "fresh")
            self.assertEqual(rem, 2_000_000)
            # partial: remaining = budget - last
            sd = root / "seed0" / "n20_hotspot"
            (sd / "n20_hotspot_seed0" / "models").mkdir(parents=True)
            (sd / "n20_hotspot_seed0" / "models" / "actor.pt").write_text("x")
            (sd / "train_stdout.log").write_text(
                "updates 1/10, total num timesteps 500000/2000000, FPS 80.\n"
            )
            st, rem = stage_status(root, 0, stage)
            self.assertEqual(st, "partial")
            self.assertEqual(rem, 1_500_000)
            # done: log reached budget
            (sd / "train_stdout.log").write_text(
                "updates 10/10, total num timesteps 2000000/2000000, FPS 80.\n"
            )
            self.assertEqual(stage_status(root, 0, stage), ("done", 0))
            # done (episode rounding): tops out ~99.9% -> still done
            (sd / "train_stdout.log").write_text(
                "updates 390/390, total num timesteps 1997824/2000000, FPS 80.\n"
            )
            self.assertEqual(stage_status(root, 0, stage)[0], "done")

    def test_branch_jobs_skip_seed_whose_trunk_failed(self):
        """P2 regression: a None init in curriculum mode means the trunk failed,
        so that seed's branches must be dropped (not warm-started off a dead
        checkpoint)."""
        branches = branch_stages(smoke=True)
        n15 = {0: Path("seed0/n15/actor.pt"), 1: None}  # seed1 trunk failed
        jobs = _plan_branch_jobs([0, 1], branches, n15, skip_trunk=False)
        self.assertEqual({seed for _, seed, _ in jobs}, {0})
        self.assertEqual(len(jobs), len(branches))  # only seed0's 4 branches

    def test_missing_trunk_actor_records_failure_not_green(self):
        """P2: a trunk that returns rc=0 but writes no actor.pt must surface a
        FAILED StageResult (so main() exits nonzero), not silently skip the
        seed's branches and exit green."""
        import src.rl.mappo_onpolicy.curriculum as cur

        with tempfile.TemporaryDirectory() as d:
            args = cur.build_parser().parse_args(
                ["--smoke", "--seeds", "0", "--parallel", "1", "--results-dir", d]
            )

            def fake_run_stage(stage, seed, results_dir, *a, **k):
                # rc=0 but never writes actor.pt -> existence check must trip.
                return cur.StageResult(
                    seed=seed,
                    stage=stage.name,
                    returncode=0,
                    elapsed_s=0.0,
                    actor_out=str(cur._actor_path(results_dir, seed, stage.name)),
                )

            orig = cur._run_stage
            cur._run_stage = fake_run_stage
            try:
                results = cur.run_curriculum(args)
            finally:
                cur._run_stage = orig

        self.assertTrue(
            [r for r in results if r.returncode != 0],
            "missing trunk actor must record a failed StageResult",
        )
        # No branch stage-runs should have been spawned for the failed seed.
        self.assertEqual([r for r in results if r.stage.startswith("n20_")], [])

    def test_branch_jobs_skip_trunk_keeps_none_init(self):
        """Under --skip-trunk a None init legitimately means 'from scratch' and
        must be kept for every seed (not confused with a trunk failure)."""
        branches = branch_stages(smoke=True)
        n15 = {0: None, 1: None}
        jobs = _plan_branch_jobs([0, 1], branches, n15, skip_trunk=True)
        self.assertEqual({seed for _, seed, _ in jobs}, {0, 1})
        self.assertTrue(all(init is None for _, _, init in jobs))
        self.assertEqual(len(jobs), 2 * len(branches))

    def test_stage_status_handles_resumed_multi_segment_log(self):
        """REGRESSION: after a resume the log holds an original segment
        (.../2000000) then a smaller remaining-budget segment; a completed
        resume (X ~= its own Y) must read as done, not partial."""
        stage = Stage("n20_hotspot", 20, "hotspot", 2_000_000, 1024)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sd = root / "seed0" / "n20_hotspot"
            (sd / "n20_hotspot_seed0" / "models").mkdir(parents=True)
            (sd / "n20_hotspot_seed0" / "models" / "actor.pt").write_text("x")
            (sd / "train_stdout.log").write_text(
                "total num timesteps 1060864/2000000, FPS 90.\n"
                "total num timesteps 937984/939136, FPS 90.\n"  # resume done
            )
            self.assertEqual(stage_status(root, 0, stage), ("done", 0))
            # interrupted mid-resume -> partial, remaining within resume budget
            (sd / "train_stdout.log").write_text(
                "total num timesteps 1060864/2000000, FPS 90.\n"
                "total num timesteps 400000/939136, FPS 90.\n"
            )
            st, rem = stage_status(root, 0, stage)
            self.assertEqual(st, "partial")
            self.assertEqual(rem, 939136 - 400000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
