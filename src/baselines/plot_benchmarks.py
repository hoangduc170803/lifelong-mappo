"""Plot Sprint 2 benchmark CSV metrics as reproducible figures.

Run from the repo root:
    python -m src.baselines.plot_benchmarks

The command writes PNG/PDF/SVG charts to:
    results/baselines/figures/
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "results" / "baselines" / "sprint2_mapf_baselines.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "baselines" / "figures"
DEFAULT_FORMATS = ("png", "pdf", "svg")

BASELINE_ORDER = [
    "prioritized_planning_default_order",
    "priority_search",
    "hungarian_prioritized_planning",
    "fifo_nearest_prioritized_planning",
    "opentcs_naive",
]

BASELINE_LABELS = {
    "prioritized_planning_default_order": "PP default",
    "priority_search": "PP-32",
    "hungarian_prioritized_planning": "Hungarian + PP",
    "fifo_nearest_prioritized_planning": "FIFO-nearest + PP",
    "opentcs_naive": "OpenTCS naive",
}

NUMERIC_COLUMNS = [
    "seed",
    "num_agents",
    "success_rate",
    "instance_makespan",
    "lower_bound_steps",
    "lower_bound_distance",
    "makespan_over_lower_bound",
    "sum_of_costs",
    "waiting_time_agent_steps",
    "throughput_tasks_per_1000_steps",
    "conflicts_total",
    "elapsed_assignment_s",
    "elapsed_planner_s",
    "elapsed_s",
]

REQUIRED_COLUMNS = {
    "baseline",
    "num_agents",
    "success",
    "instance_makespan",
    "waiting_time_agent_steps",
    "throughput_tasks_per_1000_steps",
    "elapsed_s",
}


@dataclass(frozen=True)
class ChartSpec:
    stem: str
    value_column: str
    error_column: str | None
    title: str
    ylabel: str
    scale: float = 1.0
    ylim: tuple[float, float] | None = None
    missing_label: str | None = None


CHARTS = [
    ChartSpec(
        stem="success_rate",
        value_column="success_rate",
        error_column=None,
        title="Sprint 2 Baseline Success Rate",
        ylabel="Success rate (%)",
        scale=100.0,
        ylim=(0.0, 105.0),
    ),
    ChartSpec(
        stem="makespan_success",
        value_column="instance_makespan_mean",
        error_column="instance_makespan_std",
        title="Makespan on Successful Runs",
        ylabel="Makespan (steps)",
        missing_label="n=0",
    ),
    ChartSpec(
        stem="waiting_time_success",
        value_column="waiting_time_agent_steps_mean",
        error_column="waiting_time_agent_steps_std",
        title="Waiting Time on Successful Runs",
        ylabel="Waiting time (agent-steps)",
        missing_label="n=0",
    ),
    ChartSpec(
        stem="throughput_success",
        value_column="throughput_tasks_per_1000_steps_mean",
        error_column="throughput_tasks_per_1000_steps_std",
        title="Throughput on Successful Runs",
        ylabel="Tasks / 1000 steps",
        missing_label="n=0",
    ),
    ChartSpec(
        stem="elapsed_runtime",
        value_column="elapsed_s_mean",
        error_column="elapsed_s_std",
        title="Planner Runtime",
        ylabel="Elapsed time (seconds)",
    ),
]


def read_benchmark_csv(path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
    """Read and normalize a Sprint 2 benchmark CSV."""
    df = pd.read_csv(path)
    return normalize_benchmark_df(df)


def normalize_benchmark_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with expected booleans, numerics, and baseline ordering."""
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Benchmark CSV is missing required columns: {', '.join(missing)}")

    normalized = df.copy()
    normalized["success"] = _coerce_bool_series(normalized["success"])

    for column in NUMERIC_COLUMNS:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    known = [baseline for baseline in BASELINE_ORDER if baseline in set(normalized["baseline"])]
    extras = sorted(set(normalized["baseline"]) - set(known))
    normalized["baseline"] = pd.Categorical(
        normalized["baseline"],
        categories=known + extras,
        ordered=True,
    )
    sort_columns = [column for column in ["baseline", "num_agents", "seed"] if column in normalized]
    normalized = normalized.sort_values(sort_columns, kind="stable")
    return normalized


def summarize_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize raw benchmark rows by baseline and agent count.

    Success-rate and runtime use all runs. Makespan, waiting time, and
    throughput use successful runs only, so failed/timeout rows do not distort
    the figures intended for method comparison.
    """
    normalized = normalize_benchmark_df(df)
    keys = ["baseline", "num_agents"]

    all_runs = (
        normalized.groupby(keys, observed=True)
        .agg(
            runs=("success", "size"),
            success_rate=("success", "mean"),
            elapsed_s_mean=("elapsed_s", _finite_mean),
            elapsed_s_std=("elapsed_s", _finite_std),
        )
        .reset_index()
    )

    successful = normalized[normalized["success"]].copy()
    success_metrics = (
        successful.groupby(keys, observed=True)
        .agg(
            instance_makespan_mean=("instance_makespan", _finite_mean),
            instance_makespan_std=("instance_makespan", _finite_std),
            waiting_time_agent_steps_mean=("waiting_time_agent_steps", _finite_mean),
            waiting_time_agent_steps_std=("waiting_time_agent_steps", _finite_std),
            throughput_tasks_per_1000_steps_mean=(
                "throughput_tasks_per_1000_steps",
                _finite_mean,
            ),
            throughput_tasks_per_1000_steps_std=(
                "throughput_tasks_per_1000_steps",
                _finite_std,
            ),
        )
        .reset_index()
    )

    summary = all_runs.merge(success_metrics, on=keys, how="left")
    return summary.sort_values(keys, kind="stable").reset_index(drop=True)


def generate_charts(
    summary: pd.DataFrame,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    formats: Sequence[str] = DEFAULT_FORMATS,
) -> dict[str, list[Path]]:
    """Generate all Sprint 2 benchmark charts from a summary dataframe."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, list[Path]] = {}
    for chart in CHARTS:
        fig = _plot_grouped_bars(summary, chart)
        paths = _save_figure(fig, output_dir, chart.stem, formats)
        _close_figure(fig)
        outputs[chart.stem] = paths
    return outputs


def plot_from_csv(
    csv_path: str | Path = DEFAULT_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    formats: Sequence[str] = DEFAULT_FORMATS,
) -> tuple[pd.DataFrame, dict[str, list[Path]]]:
    """Read a benchmark CSV, summarize it, and write all configured charts."""
    summary = summarize_benchmarks(read_benchmark_csv(csv_path))
    return summary, generate_charts(summary, output_dir=output_dir, formats=formats)


def _plot_grouped_bars(summary: pd.DataFrame, chart: ChartSpec):
    plt = _load_pyplot()
    from matplotlib.patches import Patch

    baselines = [baseline for baseline in BASELINE_ORDER if baseline in set(summary["baseline"])]
    baselines.extend(sorted(set(summary["baseline"].astype(str)) - set(baselines)))
    agent_counts = sorted(int(value) for value in summary["num_agents"].dropna().unique())
    positions = list(range(len(agent_counts)))
    bar_width = min(0.82 / max(len(baselines), 1), 0.18)

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    colors = plt.get_cmap("tab10")
    legend_handles = []

    for idx, baseline in enumerate(baselines):
        rows = summary[summary["baseline"].astype(str) == str(baseline)]
        values = [
            _cell_value(rows, agent_count, chart.value_column, chart.scale)
            for agent_count in agent_counts
        ]
        yerr = None
        if chart.error_column is not None:
            yerr = [
                _cell_value(rows, agent_count, chart.error_column, chart.scale)
                for agent_count in agent_counts
            ]
        offsets = [pos + (idx - (len(baselines) - 1) / 2) * bar_width for pos in positions]
        baseline_color = colors(idx % 10)
        legend_handles.append(
            Patch(
                facecolor=baseline_color,
                label=BASELINE_LABELS.get(str(baseline), str(baseline)),
            )
        )

        finite_items = [
            (offset, value, yerr[pos] if yerr is not None else math.nan)
            for pos, (offset, value) in enumerate(zip(offsets, values))
            if _is_finite_number(value)
        ]
        if finite_items:
            finite_offsets = [item[0] for item in finite_items]
            finite_values = [item[1] for item in finite_items]
            finite_stds = [item[2] for item in finite_items]
            finite_yerr = (
                _asymmetric_yerr(finite_values, finite_stds)
                if chart.error_column is not None
                else None
            )
            ax.bar(
                finite_offsets,
                finite_values,
                width=bar_width,
                color=baseline_color,
                yerr=finite_yerr,
                capsize=3 if finite_yerr is not None else 0,
                linewidth=0,
            )

        if chart.missing_label is not None:
            missing_offsets = [
                offset
                for offset, value in zip(offsets, values)
                if not _is_finite_number(value)
            ]
            if missing_offsets:
                placeholder_height = _missing_bar_height(values)
                ax.bar(
                    missing_offsets,
                    [placeholder_height] * len(missing_offsets),
                    width=bar_width,
                    color="#e5e7eb",
                    edgecolor="#6b7280",
                    hatch="//",
                    linewidth=0.5,
                )
                for offset in missing_offsets:
                    ax.text(
                        offset,
                        placeholder_height,
                        chart.missing_label,
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="#6b7280",
                    )

    ax.set_title(chart.title, fontsize=13)
    ax.set_xlabel("Number of AGVs")
    ax.set_ylabel(chart.ylabel)
    ax.set_xticks(positions, [str(value) for value in agent_counts])
    if chart.ylim is not None:
        ax.set_ylim(*chart.ylim)
    ax.grid(axis="y", color="#d1d5db", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout()
    return fig


def _save_figure(fig, output_dir: Path, stem: str, formats: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for fmt in formats:
        suffix = fmt.lower().lstrip(".")
        path = output_dir / f"{stem}.{suffix}"
        save_kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            save_kwargs["dpi"] = 180
        fig.savefig(path, format=suffix, **save_kwargs)
        paths.append(path)
    return paths


def _cell_value(rows: pd.DataFrame, agent_count: int, column: str, scale: float) -> float:
    match = rows[rows["num_agents"] == agent_count]
    if match.empty:
        return math.nan
    value = match.iloc[0][column]
    if not _is_finite_number(value):
        return math.nan
    return float(value) * scale


def _asymmetric_yerr(values: Sequence[float], stds: Sequence[float]) -> list[list[float]]:
    """Return lower/upper y-error arrays clipped at zero for non-negative metrics."""
    lowers: list[float] = []
    uppers: list[float] = []
    for value, std in zip(values, stds):
        if not _is_finite_number(value) or not _is_finite_number(std):
            lowers.append(0.0)
            uppers.append(0.0)
            continue
        as_value = float(value)
        as_std = float(std)
        lowers.append(min(as_std, as_value))
        uppers.append(as_std)
    return [lowers, uppers]


def _missing_bar_height(values: Sequence[float]) -> float:
    finite_values = [float(value) for value in values if _is_finite_number(value)]
    if not finite_values:
        return 1.0
    return max(max(finite_values) * 0.015, 0.5)


def _is_finite_number(value: object) -> bool:
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(as_float)


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _finite_mean(values: Iterable[object]) -> float:
    finite = _finite_values(values)
    return sum(finite) / len(finite) if finite else math.nan


def _finite_std(values: Iterable[object]) -> float:
    finite = _finite_values(values)
    if len(finite) < 2:
        return math.nan
    mean = sum(finite) / len(finite)
    variance = sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)
    return math.sqrt(variance)


def _finite_values(values: Iterable[object]) -> list[float]:
    finite: list[float] = []
    for value in values:
        try:
            as_float = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(as_float):
            finite.append(as_float)
    return finite


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _close_figure(fig) -> None:
    plt = _load_pyplot()
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=list(DEFAULT_FORMATS),
        help="Figure formats to export, for example: png pdf svg",
    )
    args = parser.parse_args(argv)

    summary, outputs = plot_from_csv(args.csv, args.out_dir, formats=args.formats)
    print("Sprint 2 benchmark visualization complete")
    print(f"  Rows summarized: {len(summary)}")
    print(f"  Output directory: {args.out_dir}")
    for stem, paths in outputs.items():
        print(f"  {stem}:")
        for path in paths:
            print(f"    - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
