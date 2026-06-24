"""Unit tests for the mechanism-probe aggregation (actor-free, no checkpoint).

Covers the pure functions that turn per-step idle/pending/in_flight series into
the supply-vs-execution signals and the descriptive regime label.
"""

from __future__ import annotations

import math

from src.rl.mappo_onpolicy.mechanism_probe import _episode_stats, _regime


def test_supply_limited_episode_signals():
    # Slack: agents idle, queue tiny relative to cap, almost all work cleared.
    h = 100
    s = _episode_stats(
        idle_series=[5] * h,
        pending_series=[2] * h,
        inflight_series=[3] * h,
        completions=95,
        arrivals=100,
        max_pool=50,
    )
    assert s["mean_idle"] == 5
    assert s["mean_pending"] == 2
    assert s["mean_pool_occ"] == (2 + 3) / 50  # 0.1
    assert s["clear_ratio"] == 0.95
    assert abs(s["pending_slope"]) < 1e-9  # flat queue


def test_execution_limited_episode_signals():
    # Saturated: zero idle, queue pinned at the cap, large backlog uncleared.
    h = 100
    s = _episode_stats(
        idle_series=[0] * h,
        pending_series=[45] * h,
        inflight_series=[5] * h,
        completions=50,
        arrivals=120,
        max_pool=50,
    )
    assert s["mean_idle"] == 0
    assert s["mean_pool_occ"] == 1.0  # pinned at cap (tasks being dropped)
    assert s["clear_ratio"] == 50 / 120  # < 0.5, big backlog


def test_pending_slope_detects_growing_backlog():
    growing = list(range(100))  # 0,1,...,99
    s = _episode_stats(
        idle_series=[0] * 100, pending_series=growing,
        inflight_series=[0] * 100, completions=10, arrivals=110, max_pool=50,
    )
    assert s["pending_slope"] > 0.9  # ~1 task/step growth


def test_zero_arrivals_clear_ratio_is_nan_not_zerodiv():
    s = _episode_stats(
        idle_series=[20] * 10, pending_series=[0] * 10,
        inflight_series=[0] * 10, completions=0, arrivals=0, max_pool=50,
    )
    assert math.isnan(s["clear_ratio"])  # no division by zero


def test_regime_labels_supply_vs_execution():
    supply = {"clear_ratio": 0.90, "mean_pool_occ": 0.15, "agent_util": 0.80}
    execution = {"clear_ratio": 0.55, "mean_pool_occ": 0.90, "agent_util": 0.99}
    assert _regime(supply).startswith("SUPPLY")
    assert _regime(execution).startswith("EXECUTION")


def test_regime_execution_triggers_on_high_occupancy_alone():
    # Even with a moderate clear ratio, a pool pinned near the cap is execution-
    # limited (the cap is dropping arrivals).
    capped = {"clear_ratio": 0.80, "mean_pool_occ": 0.70, "agent_util": 0.9}
    assert _regime(capped).startswith("EXECUTION")
