"""Visualize the OpenTCS warehouse map and verify graph topology.

Run from the repo root:
    python -m src.utils.visualize

The command writes:
    results/map/warehouse_full.png
    results/map/warehouse_largest_scc.png
    results/map/topology_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import networkx as nx

from src.map_parser import graph_summary, parse_opentcs_map


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAP_FILE = REPO_ROOT / "warehouse_map.xml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "map"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _top_n(counter: Counter, n: int = 10) -> list[dict[str, int]]:
    return [{"value": str(value), "count": int(count)} for value, count in counter.most_common(n)]


def _component_sizes(components: list[list[str]], limit: int = 10) -> list[int]:
    return sorted((len(c) for c in components), reverse=True)[:limit]


def _sorted_components(components) -> list[list[str]]:
    sorted_node_lists = [sorted(component) for component in components]
    return sorted(
        sorted_node_lists,
        key=lambda nodes: (-len(nodes), nodes[0] if nodes else ""),
    )


def _has_reverse_edge(G: nx.DiGraph, u: str, v: str) -> bool:
    return G.has_edge(v, u)


def _raw_xml_checks(xml_path: Path) -> dict[str, Any]:
    """Validate XML-level references before NetworkX can hide them."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    point_names: list[str] = []
    bad_coordinate_points: list[str] = []
    for point in root.findall("point"):
        name = point.get("name", "")
        point_names.append(name)
        for attr in ("xPosition", "yPosition", "zPosition"):
            try:
                value = float(point.get(attr, "0"))
            except ValueError:
                bad_coordinate_points.append(name)
                break
            if not math.isfinite(value):
                bad_coordinate_points.append(name)
                break

    point_counts = Counter(point_names)
    duplicate_points = sorted(name for name, count in point_counts.items() if count > 1)
    point_set = set(point_names)

    path_names: list[str] = []
    locked_paths = 0
    missing_endpoint_paths: list[dict[str, str | None]] = []
    self_loop_paths: list[str] = []
    nonpositive_length_paths: list[str] = []
    duplicate_directed_edges: Counter[tuple[str | None, str | None]] = Counter()

    for path in root.findall("path"):
        name = path.get("name", "")
        src = path.get("sourcePoint")
        dst = path.get("destinationPoint")
        path_names.append(name)

        if path.get("locked", "false").lower() == "true":
            locked_paths += 1
            continue

        if src not in point_set or dst not in point_set:
            missing_endpoint_paths.append({"path": name, "source": src, "destination": dst})
        if src == dst:
            self_loop_paths.append(name)

        try:
            length = float(path.get("length", "0"))
        except ValueError:
            length = 0.0
        if length <= 0:
            nonpositive_length_paths.append(name)

        duplicate_directed_edges[(src, dst)] += 1

    path_counts = Counter(path_names)
    duplicate_paths = sorted(name for name, count in path_counts.items() if count > 1)
    duplicate_edges = [
        {"source": src, "destination": dst, "count": int(count)}
        for (src, dst), count in duplicate_directed_edges.items()
        if count > 1
    ]

    return {
        "xml_path": xml_path,
        "points": len(point_names),
        "paths": len(path_names),
        "locked_paths_skipped_by_parser": locked_paths,
        "duplicate_point_names": duplicate_points,
        "duplicate_path_names": duplicate_paths,
        "duplicate_directed_edges": duplicate_edges,
        "missing_endpoint_paths": missing_endpoint_paths,
        "self_loop_paths": self_loop_paths,
        "nonpositive_length_paths": nonpositive_length_paths,
        "bad_coordinate_points": sorted(set(bad_coordinate_points)),
    }


def verify_topology(G: nx.DiGraph, xml_path: str | Path | None = None) -> dict[str, Any]:
    """Return a serializable topology report for a directed warehouse graph."""
    report: dict[str, Any] = {"graph": graph_summary(G)}
    if xml_path is not None:
        report["xml"] = _raw_xml_checks(Path(xml_path))

    nodes = list(G.nodes())
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    total_degrees = {
        node: in_degrees.get(node, 0) + out_degrees.get(node, 0)
        for node in nodes
    }

    weak_components = _sorted_components(nx.weakly_connected_components(G))
    strong_components = _sorted_components(nx.strongly_connected_components(G))
    largest_scc = strong_components[0] if strong_components else []
    largest_scc_size = len(largest_scc)
    largest_wcc_size = len(weak_components[0]) if weak_components else 0
    sccs_excluded_from_largest = [
        {"size": len(component), "nodes": component}
        for component in strong_components[1:]
    ]

    isolated = [n for n in nodes if total_degrees[n] == 0]
    sources = [n for n in nodes if in_degrees[n] == 0 and out_degrees[n] > 0]
    sinks = [n for n in nodes if out_degrees[n] == 0 and in_degrees[n] > 0]
    one_way_edges = [(u, v) for u, v in G.edges() if not _has_reverse_edge(G, u, v)]

    edge_lengths = [
        float(data.get("length", data.get("weight", 0.0)))
        for _, _, data in G.edges(data=True)
    ]
    max_length = max(edge_lengths) if edge_lengths else 0.0
    min_length = min(edge_lengths) if edge_lengths else 0.0

    report["topology"] = {
        "weak_component_count": len(weak_components),
        "weak_component_sizes_top10": _component_sizes(weak_components),
        "strong_component_count": len(strong_components),
        "strong_component_sizes_top10": _component_sizes(strong_components),
        "sccs_excluded_from_largest": sccs_excluded_from_largest,
        "largest_wcc_ratio": largest_wcc_size / len(nodes) if nodes else 0.0,
        "largest_scc_ratio": largest_scc_size / len(nodes) if nodes else 0.0,
        "isolated_nodes": len(isolated),
        "source_nodes": len(sources),
        "sink_nodes": len(sinks),
        "one_way_edges": len(one_way_edges),
        "bidirectional_edge_pairs": (G.number_of_edges() - len(one_way_edges)) // 2,
        "degree_distribution_top10": _top_n(Counter(total_degrees.values())),
        "in_degree_distribution_top10": _top_n(Counter(in_degrees.values())),
        "out_degree_distribution_top10": _top_n(Counter(out_degrees.values())),
        "edge_length_min": min_length,
        "edge_length_max": max_length,
    }

    report["samples"] = {
        "isolated_nodes": isolated[:20],
        "source_nodes": sources[:20],
        "sink_nodes": sinks[:20],
        "one_way_edges": [{"source": u, "destination": v} for u, v in one_way_edges[:20]],
    }
    report["verdict"] = _build_verdict(report)
    return report


def _build_verdict(report: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []

    xml_report = report.get("xml", {})
    if xml_report.get("missing_endpoint_paths"):
        failures.append("XML has paths whose source/destination point is missing.")
    if xml_report.get("duplicate_point_names"):
        failures.append("XML has duplicate point names.")
    if xml_report.get("bad_coordinate_points"):
        failures.append("XML has points with non-finite coordinates.")
    if xml_report.get("nonpositive_length_paths"):
        failures.append("XML has unlocked paths with non-positive length.")
    if xml_report.get("self_loop_paths"):
        warnings.append("XML has self-loop paths.")
    if xml_report.get("duplicate_directed_edges"):
        warnings.append("XML has duplicate directed edges.")

    topology = report["topology"]
    if topology["isolated_nodes"] > 0:
        warnings.append("Graph contains isolated nodes.")
    if topology["source_nodes"] > 0:
        warnings.append("Graph contains source-only nodes.")
    if topology["sink_nodes"] > 0:
        warnings.append("Graph contains sink-only nodes.")
    if not report["graph"]["is_strongly_connected"]:
        warnings.append("Directed graph is not strongly connected; use largest SCC for MAPF training.")

    return {
        "ok_for_largest_scc_training": not failures,
        "failures": failures,
        "warnings": warnings,
    }


def _node_colors(G: nx.DiGraph, color_by: str) -> tuple[list[str], dict[str, str]]:
    if color_by == "type":
        colors = []
        for _, data in G.nodes(data=True):
            colors.append("#d94841" if data.get("is_halt") else "#2f6fbb")
        return colors, {"HALT_POSITION": "#d94841", "OTHER": "#2f6fbb"}

    if color_by == "component":
        components = _sorted_components(nx.weakly_connected_components(G))
        cmap = plt.get_cmap("tab20", max(len(components), 1))
        palette = [to_hex(cmap(i)) for i in range(len(components))]
        comp_idx = {}
        for idx, comp in enumerate(components):
            for node in comp:
                comp_idx[node] = idx
        colors = [palette[comp_idx[node]] for node in G.nodes()]
        return (
            colors,
            {
                f"WCC {i + 1} (n={len(component)})": palette[i]
                for i, component in enumerate(components)
            },
        )

    raise ValueError(f"Unsupported color mode: {color_by}")


def _topology_marker_groups(G: nx.DiGraph) -> list[tuple[str, list[str], str, str, int]]:
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    isolated = [
        node
        for node in G.nodes()
        if in_degrees.get(node, 0) == 0 and out_degrees.get(node, 0) == 0
    ]
    sources = [
        node
        for node in G.nodes()
        if in_degrees.get(node, 0) == 0 and out_degrees.get(node, 0) > 0
    ]
    sinks = [
        node
        for node in G.nodes()
        if out_degrees.get(node, 0) == 0 and in_degrees.get(node, 0) > 0
    ]
    return [
        ("isolated", isolated, "x", "#d62728", 64),
        ("source-only", sources, "^", "#ff7f0e", 42),
        ("sink-only", sinks, "s", "#6f42c1", 38),
    ]


def plot_map(
    G: nx.DiGraph,
    output_path: str | Path,
    title: str,
    color_by: str = "type",
    highlight_topology: bool = False,
    show_labels: bool = False,
    dpi: int = 180,
) -> Path:
    """Render a directed graph map to a PNG with matplotlib."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pos = {node: (data["x"], data["y"]) for node, data in G.nodes(data=True)}
    node_colors, legend_items = _node_colors(G, color_by)

    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    width = max(xs) - min(xs) if xs else 1.0
    height = max(ys) - min(ys) if ys else 1.0
    aspect = max(width / max(height, 1e-9), 1.0)
    fig_width = min(max(8.0, 8.0 * aspect), 18.0)
    fig_height = min(max(6.0, fig_width / aspect), 12.0)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    nx.draw_networkx_edges(
        G,
        pos=pos,
        ax=ax,
        edge_color="#8b949e",
        arrows=True,
        arrowsize=3,
        arrowstyle="-|>",
        width=0.35,
        alpha=0.42,
        min_source_margin=0,
        min_target_margin=0,
        connectionstyle="arc3,rad=0.02",
    )
    nx.draw_networkx_nodes(
        G,
        pos=pos,
        ax=ax,
        node_color=node_colors,
        node_size=5,
        linewidths=0,
        alpha=0.9,
    )

    if highlight_topology:
        for label, group_nodes, marker, color, size in _topology_marker_groups(G):
            if not group_nodes:
                continue
            group_x = [pos[node][0] for node in group_nodes]
            group_y = [pos[node][1] for node in group_nodes]
            ax.scatter(
                group_x,
                group_y,
                s=size,
                marker=marker,
                c=color,
                linewidths=1.4,
                label=f"{label} ({len(group_nodes)})",
                zorder=5,
            )

    if show_labels:
        nx.draw_networkx_labels(G, pos=pos, ax=ax, font_size=3)

    for label, color in legend_items.items():
        ax.scatter([], [], s=16, color=color, label=label)
    legend_count = len(legend_items) + (3 if highlight_topology else 0)
    # Keep the legend outside the plotting area so topology markers near the
    # map boundary remain visible.
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        fontsize=6 if legend_count > 10 else 7,
        frameon=False,
        ncol=1,
    )
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("xPosition")
    ax.set_ylabel("yPosition")
    ax.grid(True, color="#e5e7eb", linewidth=0.3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run(
    xml_path: str | Path = DEFAULT_MAP_FILE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    show_labels: bool = False,
) -> dict[str, Any]:
    xml_path = Path(xml_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_graph = parse_opentcs_map(xml_path)
    scc_graph = parse_opentcs_map(xml_path, restrict_to_largest_scc=True)

    full_report = verify_topology(full_graph, xml_path=xml_path)
    scc_report = verify_topology(scc_graph)
    combined_report = {
        "map_file": xml_path,
        "full_graph": full_report,
        "largest_scc": scc_report,
    }

    full_png = plot_map(
        full_graph,
        output_dir / "warehouse_full.png",
        "OpenTCS Warehouse Map - Full Graph",
        color_by="component",
        highlight_topology=True,
        show_labels=show_labels,
    )
    scc_png = plot_map(
        scc_graph,
        output_dir / "warehouse_largest_scc.png",
        "OpenTCS Warehouse Map - Largest Strongly Connected Component",
        color_by="type",
        show_labels=show_labels,
    )

    report_path = output_dir / "topology_report.json"
    combined_report["outputs"] = {
        "full_png": full_png,
        "largest_scc_png": scc_png,
        "report_json": report_path,
    }
    report_path.write_text(
        json.dumps(combined_report, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return combined_report


def _print_report(report: dict[str, Any]) -> None:
    full = report["full_graph"]
    scc = report["largest_scc"]
    outputs = report["outputs"]

    print("Map visualization/topology complete")
    print(f"  Full graph: {full['graph']['num_nodes']} nodes, {full['graph']['num_edges']} edges")
    print(
        "  Full topology: "
        f"{full['topology']['weak_component_count']} WCC, "
        f"{full['topology']['strong_component_count']} SCC, "
        f"largest SCC ratio={full['topology']['largest_scc_ratio']:.3f}"
    )
    print(
        "  Largest SCC: "
        f"{scc['graph']['num_nodes']} nodes, {scc['graph']['num_edges']} edges, "
        f"strongly_connected={scc['graph']['is_strongly_connected']}"
    )
    print(f"  Verdict: ok_for_largest_scc_training={full['verdict']['ok_for_largest_scc_training']}")
    if full["verdict"]["warnings"]:
        print("  Warnings:")
        for warning in full["verdict"]["warnings"]:
            print(f"    - {warning}")
    print("  Outputs:")
    for name, path in outputs.items():
        print(f"    - {name}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_MAP_FILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--show-labels", action="store_true")
    args = parser.parse_args()

    report = run(args.xml, args.out_dir, show_labels=args.show_labels)
    _print_report(report)


if __name__ == "__main__":
    main()
