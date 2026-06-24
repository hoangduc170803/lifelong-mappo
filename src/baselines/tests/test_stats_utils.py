"""Tests for shared baseline statistics helpers."""

from __future__ import annotations

import unittest

from src.baselines.stats_utils import paired_comparison, welch_comparison, wilson_ci


class TestWilsonCI(unittest.TestCase):
    def test_known_value_half_at_n10(self):
        # 5/10 with z=1.96: textbook Wilson interval ~ (0.237, 0.763).
        lo, hi = wilson_ci(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)

    def test_zero_successes_lower_bound_is_zero(self):
        lo, hi = wilson_ci(0, 10)
        self.assertEqual(lo, 0.0)
        # Wilson upper bound for 0/10 is ~0.278 — strictly positive,
        # which is exactly why it beats the normal approximation here.
        self.assertAlmostEqual(hi, 0.2775, places=3)

    def test_all_successes_upper_bound_is_one(self):
        lo, hi = wilson_ci(10, 10)
        self.assertEqual(hi, 1.0)
        self.assertAlmostEqual(lo, 0.7225, places=3)

    def test_symmetry_complement(self):
        lo_a, hi_a = wilson_ci(3, 10)
        lo_b, hi_b = wilson_ci(7, 10)
        self.assertAlmostEqual(lo_a, 1.0 - hi_b, places=9)
        self.assertAlmostEqual(hi_a, 1.0 - lo_b, places=9)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            wilson_ci(1, 0)
        with self.assertRaises(ValueError):
            wilson_ci(-1, 10)
        with self.assertRaises(ValueError):
            wilson_ci(11, 10)


class TestPairedComparison(unittest.TestCase):
    def test_consistent_positive_effect_is_significant(self):
        # Treatment beats baseline on every seed by ~constant margin.
        baseline = [50.0, 49.0, 52.0, 51.0, 50.5, 49.5, 52.5, 50.0, 51.5, 49.0]
        treatment = [v + 3.0 for v in baseline]
        out = paired_comparison(treatment, baseline)
        self.assertEqual(out["n_pairs"], 10)
        self.assertEqual(out["wins"], 10)
        self.assertAlmostEqual(out["mean_diff"], 3.0, places=6)
        self.assertLess(out["paired_t_p"], 0.05)
        self.assertLess(out["wilcoxon_p"], 0.05)
        lo, hi = out["bootstrap_ci95_mean_diff"]
        self.assertGreater(lo, 0.0)  # CI excludes 0 -> significant
        self.assertGreaterEqual(hi, lo)

    def test_bootstrap_is_deterministic_given_seed(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [0.0, 1.0, 2.0, 3.0, 4.0]
        self.assertEqual(
            paired_comparison(a, b)["bootstrap_ci95_mean_diff"],
            paired_comparison(a, b)["bootstrap_ci95_mean_diff"],
        )

    def test_mismatched_length_raises(self):
        with self.assertRaises(ValueError):
            paired_comparison([1.0, 2.0], [1.0])


class TestWelchComparison(unittest.TestCase):
    def test_separated_samples_are_significant(self):
        a = [120.0, 125.0, 130.0, 128.0, 127.0, 132.0]
        b = [80.0, 85.0, 90.0, 84.0, 86.0, 88.0]
        out = welch_comparison(a, b)
        self.assertGreater(out["mean_diff"], 0.0)
        self.assertLess(out["welch_p"], 0.01)

    def test_overlapping_samples_not_significant(self):
        a = [100.0, 101.0, 99.0, 100.5, 99.5]
        b = [100.2, 99.8, 100.1, 99.9, 100.0]
        self.assertGreater(welch_comparison(a, b)["welch_p"], 0.05)

    def test_too_small_raises(self):
        with self.assertRaises(ValueError):
            welch_comparison([1.0], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
