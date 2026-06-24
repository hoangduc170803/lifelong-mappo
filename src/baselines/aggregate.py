"""Aggregate Sprint 2 benchmark CSV rows into a compact summary table."""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable, Sequence


def aggregate_rows(
    rows: Iterable[dict[str, str]],
    ci_method: str = "none",
    ci_level: float = 0.95,
    bootstrap_samples: int = 2000,
    seed: int = 0,
    failure_makespan_penalty: float = 512.0,
) -> list[dict[str, object]]:
    """Group raw benchmark rows by baseline and agent count."""
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["baseline"], int(row["num_agents"]))].append(row)

    summary: list[dict[str, object]] = []
    for (baseline, num_agents), items in sorted(grouped.items()):
        successes = [_as_bool(row["success"]) for row in items]
        successful_rows = [row for row, ok in zip(items, successes) if ok]
        failure_reasons = Counter(
            row.get("failure_reason", "") or "unspecified"
            for row, ok in zip(items, successes)
            if not ok
        )
        max_time = failure_makespan_penalty
        makespan_all = _penalized_values(items, "instance_makespan", max_time)
        wait_all = _penalized_values(
            items,
            "waiting_time_agent_steps",
            max_time * max(num_agents, 1),
        )
        throughput_all = [
            _float_or_default(row.get("throughput_tasks_per_1000_steps"), 0.0)
            if _as_bool(row["success"])
            else 0.0
            for row in items
        ]

        success_values = [1.0 if ok else 0.0 for ok in successes]
        success_ci = _mean_ci(
            success_values,
            ci_method,
            ci_level,
            bootstrap_samples,
            seed,
        )
        makespan_success = _finite_field_values(successful_rows, "instance_makespan")
        ratio_success = _finite_field_values(successful_rows, "makespan_over_lower_bound")
        wait_success = _finite_field_values(successful_rows, "waiting_time_agent_steps")
        elapsed_all = _finite_field_values(items, "elapsed_s")
        summary.append(
            {
                "baseline": baseline,
                "num_agents": num_agents,
                "runs": len(items),
                "success_rate": _mean(success_values),
                "success_rate_ci_low": success_ci[0],
                "success_rate_ci_high": success_ci[1],
                "mean_makespan_success": _mean(makespan_success),
                "std_makespan_success": _std(makespan_success),
                "mean_ratio_success": _mean(ratio_success),
                "std_ratio_success": _std(ratio_success),
                "mean_wait_success": _mean(wait_success),
                "std_wait_success": _std(wait_success),
                "mean_makespan_all_penalized": _mean(makespan_all),
                "std_makespan_all_penalized": _std(makespan_all),
                "mean_wait_all_penalized": _mean(wait_all),
                "std_wait_all_penalized": _std(wait_all),
                "mean_throughput_all": _mean(throughput_all),
                "std_throughput_all": _std(throughput_all),
                "mean_elapsed_s": _mean(elapsed_all),
                "failure_reasons": ", ".join(
                    f"{reason}:{count}" for reason, count in sorted(failure_reasons.items())
                ),
            }
        )
    return summary


def render_markdown(summary: Sequence[dict[str, object]]) -> str:
    """Render aggregate rows as a markdown table."""
    headers = [
        "baseline",
        "agents",
        "runs",
        "success",
        "success_ci",
        "mean_makespan_success",
        "std_makespan_success",
        "mean_ratio_success",
        "std_ratio_success",
        "mean_wait_success",
        "std_wait_success",
        "mean_makespan_all_penalized",
        "std_makespan_all_penalized",
        "mean_wait_all_penalized",
        "std_wait_all_penalized",
        "mean_throughput_all",
        "std_throughput_all",
        "mean_elapsed_s",
        "failure_reasons",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in summary:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["baseline"]),
                    str(row["num_agents"]),
                    str(row["runs"]),
                    _fmt_float(row["success_rate"]),
                    _fmt_ci(row["success_rate_ci_low"], row["success_rate_ci_high"]),
                    _fmt_float(row["mean_makespan_success"]),
                    _fmt_float(row["std_makespan_success"]),
                    _fmt_float(row["mean_ratio_success"]),
                    _fmt_float(row["std_ratio_success"]),
                    _fmt_float(row["mean_wait_success"]),
                    _fmt_float(row["std_wait_success"]),
                    _fmt_float(row["mean_makespan_all_penalized"]),
                    _fmt_float(row["std_makespan_all_penalized"]),
                    _fmt_float(row["mean_wait_all_penalized"]),
                    _fmt_float(row["std_wait_all_penalized"]),
                    _fmt_float(row["mean_throughput_all"]),
                    _fmt_float(row["std_throughput_all"]),
                    _fmt_float(row["mean_elapsed_s"], digits=4),
                    str(row["failure_reasons"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def read_csv(path: Path | str) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mean_field(rows: Sequence[dict[str, str]], field: str) -> float:
    values = _finite_field_values(rows, field)
    return _mean(values)


def _std_field(rows: Sequence[dict[str, str]], field: str) -> float:
    values = _finite_field_values(rows, field)
    return _std(values)


def _mean(values: Sequence[float]) -> float:
    return mean(values) if values else math.nan


def _std(values: Sequence[float]) -> float:
    return stdev(values) if len(values) >= 2 else math.nan


def _finite_field_values(rows: Sequence[dict[str, str]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.get(field, "")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _penalized_values(
    rows: Sequence[dict[str, str]],
    field: str,
    failure_penalty: float,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        if _as_bool(row["success"]):
            values.append(_float_or_default(row.get(field), failure_penalty))
        else:
            values.append(failure_penalty)
    return values


def _float_or_default(raw: object, default: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _mean_ci(
    values: Sequence[float],
    method: str,
    ci_level: float,
    bootstrap_samples: int,
    seed: int,
) -> tuple[float, float]:
    if not values or method == "none":
        return math.nan, math.nan
    if method == "normal":
        if len(values) < 2:
            return math.nan, math.nan
        z = 1.96 if abs(ci_level - 0.95) < 1e-9 else 1.96
        half_width = z * stdev(values) / math.sqrt(len(values))
        center = mean(values)
        return center - half_width, center + half_width
    if method == "bootstrap":
        return _bootstrap_mean_ci(values, ci_level, bootstrap_samples, seed)
    raise ValueError(f"unknown ci_method: {method}")


def _bootstrap_mean_ci(
    values: Sequence[float],
    ci_level: float,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if samples < 1:
        return math.nan, math.nan
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(samples):
        means.append(mean(values[rng.randrange(n)] for _ in range(n)))
    means.sort()
    alpha = max(0.0, min(1.0, 1.0 - ci_level))
    lo_idx = int((alpha / 2.0) * (samples - 1))
    hi_idx = int((1.0 - alpha / 2.0) * (samples - 1))
    return means[lo_idx], means[hi_idx]


def _as_bool(raw: str) -> bool:
    return raw.strip().lower() in {"true", "1", "yes"}


def _fmt_float(value: object, digits: int = 3) -> str:
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(as_float):
        return ""
    return f"{as_float:.{digits}f}"


def _fmt_ci(low: object, high: object) -> str:
    lo = _fmt_float(low)
    hi = _fmt_float(high)
    return f"[{lo}, {hi}]" if lo and hi else ""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "--ci",
        choices=["none", "normal", "bootstrap"],
        default="none",
        help="Confidence interval method for success rate.",
    )
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--failure-makespan-penalty",
        type=float,
        default=512.0,
        help="Makespan assigned to failed runs when computing all-run penalized metrics.",
    )
    args = parser.parse_args(argv)
    summary = aggregate_rows(
        read_csv(args.csv_path),
        ci_method=args.ci,
        ci_level=args.ci_level,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        failure_makespan_penalty=args.failure_makespan_penalty,
    )
    print(render_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
