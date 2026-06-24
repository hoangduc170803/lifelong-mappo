"""Parse OpenTCS XML map to networkx directed graph.

XML format:
    <point name="0001" xPosition="..." yPosition="..." type="HALT_POSITION" .../>
    <path name="0001 --- 0002" sourcePoint="0001" destinationPoint="0002"
          length="950" maxVelocity="1000" locked="false"/>
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union

import networkx as nx


def parse_opentcs_map(
    xml_path: Union[str, Path],
    restrict_to_largest_scc: bool = False,
) -> nx.DiGraph:
    """Parse OpenTCS XML map file into a directed graph.

    Nodes carry: x, y, z (float), type (str), is_halt (bool).
    Edges carry: length, max_velocity (float), weight=length.
    Locked paths are skipped.

    The full map has 14 weakly-connected components and 199 strongly-connected
    components (most are dead-end halt positions). For MAPF training pass
    `restrict_to_largest_scc=True` to keep only the largest SCC (~1313 nodes)
    where every node pair is mutually reachable.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    G = nx.DiGraph()

    for point in root.findall("point"):
        name = point.get("name")
        x = float(point.get("xPosition", 0))
        y = float(point.get("yPosition", 0))
        z = float(point.get("zPosition", 0))
        ptype = point.get("type", "")
        G.add_node(
            name,
            x=x,
            y=y,
            z=z,
            type=ptype,
            is_halt=ptype == "HALT_POSITION",
        )

    for path in root.findall("path"):
        if path.get("locked", "false").lower() == "true":
            continue
        src = path.get("sourcePoint")
        dst = path.get("destinationPoint")
        length = float(path.get("length", 0))
        max_v = float(path.get("maxVelocity", 0))
        G.add_edge(src, dst, length=length, max_velocity=max_v, weight=length)

    if restrict_to_largest_scc:
        largest_scc = max(nx.strongly_connected_components(G), key=len)
        G = G.subgraph(largest_scc).copy()

    return G


def graph_summary(G: nx.DiGraph) -> dict:
    """Quick stats for verification."""
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    halt_count = sum(1 for _, data in G.nodes(data=True) if data.get("is_halt"))
    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "halt_nodes": halt_count,
        "avg_in_degree": sum(in_degrees) / len(in_degrees) if in_degrees else 0,
        "avg_out_degree": sum(out_degrees) / len(out_degrees) if out_degrees else 0,
        "max_out_degree": max(out_degrees) if out_degrees else 0,
        "is_strongly_connected": nx.is_strongly_connected(G),
        "num_weakly_connected_components": nx.number_weakly_connected_components(G),
    }


if __name__ == "__main__":
    import sys

    xml_file = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "d:/IPPO_MAPF/warehouse_map.xml"
    )
    G = parse_opentcs_map(xml_file)
    summary = graph_summary(G)
    print(f"Map: {xml_file}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
