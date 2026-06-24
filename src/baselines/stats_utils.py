"""Shared statistics helpers for baseline runners and RL evals.

Single source for the Wilson score interval so every success-rate number in
the evidence CSVs/JSONs uses the same formula (PLAN: proportions at n <= 30
must use Wilson, never the normal approximation), plus the paired/unpaired
significance tests so the thesis-grade p-values live in the JSON artifacts
(machine-checkable provenance) rather than only in markdown prose.

The significance helpers import scipy/numpy lazily so baseline ProcessPool
workers (Windows spawn) that only need ``wilson_ci`` do not pay the import.
"""

from __future__ import annotations

import math
from typing import Sequence


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval (default z) for a binomial proportion."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    if not 0 <= successes <= n:
        raise ValueError("successes must be in [0, n]")
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def paired_comparison(
    treatment: Sequence[float],
    baseline: Sequence[float],
    n_boot: int = 10000,
    boot_seed: int = 0,
) -> dict[str, object]:
    """Paired test of ``treatment - baseline`` over seed-aligned samples.

    Returns the paired (related) t-test, the Wilcoxon signed-rank test, the
    mean paired difference, the count of seeds where treatment > baseline, and a
    percentile bootstrap 95% CI of the mean difference. Inputs must be the same
    length and aligned by seed (index i = same seed for both).
    """
    import numpy as np
    from scipy import stats as _ss

    a = np.asarray(treatment, dtype=float)
    b = np.asarray(baseline, dtype=float)
    if a.shape != b.shape or a.size < 2:
        raise ValueError("treatment and baseline must be aligned, length >= 2")
    diff = a - b
    t_stat, t_p = _ss.ttest_rel(a, b)
    try:  # Wilcoxon errors if all diffs are zero
        w_stat, w_p = _ss.wilcoxon(a, b)
        w_stat, w_p = float(w_stat), float(w_p)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    rng = np.random.default_rng(boot_seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    boot_means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {
        "n_pairs": int(diff.size),
        "mean_treatment": float(a.mean()),
        "mean_baseline": float(b.mean()),
        "mean_diff": float(diff.mean()),
        "wins": int((diff > 0).sum()),
        "paired_t_stat": float(t_stat),
        "paired_t_p": float(t_p),
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
        "bootstrap_ci95_mean_diff": [float(lo), float(hi)],
    }


def welch_comparison(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
) -> dict[str, object]:
    """Welch's unequal-variance t-test of ``sample_a - sample_b`` (unpaired)."""
    import numpy as np
    from scipy import stats as _ss

    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    if a.size < 2 or b.size < 2:
        raise ValueError("each sample must have length >= 2")
    t_stat, t_p = _ss.ttest_ind(a, b, equal_var=False)
    return {
        "n_a": int(a.size),
        "n_b": int(b.size),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(a.mean() - b.mean()),
        "welch_t": float(t_stat),
        "welch_p": float(t_p),
    }
