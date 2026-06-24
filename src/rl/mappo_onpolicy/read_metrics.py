"""Read tensorboardX scalar logs and emit a CSV summary.

The warehouse runner writes one ``events.out.tfevents.*`` file per scalar
tag under ``<run_dir>/logs/<tag>/<tag>/``. This script walks the log
directory, decodes each event, and aggregates step/value pairs into both
a per-tag CSV and a JSON summary with first/last/mean/min/max values.

The aggregation is deliberately framework-light: only ``tensorboardX``
(already pinned) and the stdlib. Useful for thesis tables and the
Sprint 3 vs on-policy comparison without standing up a tensorboard server.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from tensorboardX.proto import event_pb2


def _read_tfevents(path: Path) -> Iterator[event_pb2.Event]:
    """Yield decoded protobuf events from a tfevents file."""
    with path.open("rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                return
            (length,) = struct.unpack("<Q", header)
            _ = f.read(4)  # masked CRC of length; ignore
            payload = f.read(length)
            _ = f.read(4)  # masked CRC of payload; ignore
            event = event_pb2.Event()
            event.ParseFromString(payload)
            yield event


@dataclass
class ScalarSeries:
    tag: str
    steps: list[int]
    values: list[float]

    def summary(self) -> dict[str, float]:
        if not self.values:
            return {"tag": self.tag, "n": 0}
        arr = np.asarray(self.values, dtype=np.float64)
        return {
            "tag": self.tag,
            "n": int(arr.size),
            "first": float(arr[0]),
            "last": float(arr[-1]),
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "first_step": int(self.steps[0]),
            "last_step": int(self.steps[-1]),
        }


def collect_series(log_dir: Path) -> dict[str, ScalarSeries]:
    """Walk a runner log directory and return scalar series keyed by tag."""
    series: dict[str, ScalarSeries] = {}
    for event_file in sorted(log_dir.rglob("events.out.tfevents.*")):
        for event in _read_tfevents(event_file):
            if not event.HasField("summary"):
                continue
            for value in event.summary.value:
                if not value.HasField("simple_value"):
                    continue
                tag = value.tag
                s = series.setdefault(tag, ScalarSeries(tag=tag, steps=[], values=[]))
                s.steps.append(int(event.step))
                s.values.append(float(value.simple_value))

    for s in series.values():
        pairs = sorted(zip(s.steps, s.values), key=lambda kv: kv[0])
        s.steps = [step for step, _ in pairs]
        s.values = [value for _, value in pairs]
    return series


def write_csv(series: dict[str, ScalarSeries], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[int, str, float]] = []
    for s in series.values():
        for step, value in zip(s.steps, s.values):
            rows.append((step, s.tag, value))
    rows.sort(key=lambda row: (row[0], row[1]))
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "tag", "value"])
        writer.writerows(rows)


def write_summary(series: dict[str, ScalarSeries], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {tag: s.summary() for tag, s in sorted(series.items())}
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        type=Path,
        required=True,
        help="Path to <run_dir>/logs/",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Where to write the long-form CSV (default: <log-dir>/../metrics.csv)",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Where to write per-tag JSON summary (default: <log-dir>/../summary.json)",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=None,
        help="Optional subset of tags to print to stdout.",
    )
    args = parser.parse_args(argv)

    log_dir = args.log_dir.resolve()
    if not log_dir.is_dir():
        parser.error(f"--log-dir not found: {log_dir}")

    series = collect_series(log_dir)
    if not series:
        print(f"No scalar events found under {log_dir}", flush=True)
        return 1

    csv_out = args.csv_out or log_dir.parent / "metrics.csv"
    summary_out = args.summary_out or log_dir.parent / "summary.json"
    write_csv(series, csv_out)
    write_summary(series, summary_out)
    print(f"Wrote {csv_out} ({sum(len(s.values) for s in series.values())} rows)")
    print(f"Wrote {summary_out}")

    requested = set(args.tags) if args.tags else None
    for tag in sorted(series):
        if requested is not None and tag not in requested:
            continue
        summ = series[tag].summary()
        print(
            f"  {tag:<60s} n={summ['n']:>5d}"
            f" first={summ['first']:+.4g} last={summ['last']:+.4g}"
            f" mean={summ['mean']:+.4g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
