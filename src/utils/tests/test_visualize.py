"""Tests for map visualization and topology verification helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import networkx as nx

from src.utils.visualize import DEFAULT_MAP_FILE, plot_map, verify_topology
from src.map_parser import parse_opentcs_map


class TestTopologyVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full_graph = parse_opentcs_map(DEFAULT_MAP_FILE)
        cls.scc_graph = parse_opentcs_map(DEFAULT_MAP_FILE, restrict_to_largest_scc=True)

    def test_full_map_expected_component_shape(self):
        report = verify_topology(self.full_graph, xml_path=DEFAULT_MAP_FILE)
        self.assertEqual(report["graph"]["num_nodes"], 1511)
        self.assertEqual(report["graph"]["num_edges"], 2466)
        self.assertEqual(report["topology"]["weak_component_count"], 14)
        self.assertEqual(report["topology"]["strong_component_count"], 199)
        excluded = report["topology"]["sccs_excluded_from_largest"]
        self.assertEqual(len(excluded), 198)
        self.assertEqual(sum(component["size"] for component in excluded), 198)
        self.assertGreaterEqual(report["topology"]["largest_scc_ratio"], 0.85)
        self.assertTrue(report["verdict"]["ok_for_largest_scc_training"])

    def test_largest_scc_is_strongly_connected(self):
        report = verify_topology(self.scc_graph)
        self.assertEqual(report["graph"]["num_nodes"], 1313)
        self.assertTrue(report["graph"]["is_strongly_connected"])
        self.assertEqual(report["topology"]["strong_component_count"], 1)
        self.assertEqual(report["topology"]["sccs_excluded_from_largest"], [])
        self.assertEqual(report["topology"]["isolated_nodes"], 0)

    def test_plot_map_writes_png(self):
        G = nx.DiGraph()
        G.add_node("a", x=0.0, y=0.0, is_halt=True, type="HALT_POSITION")
        G.add_node("b", x=1.0, y=0.0, is_halt=False, type="POINT")
        G.add_edge("a", "b", length=1.0, weight=1.0)

        with tempfile.TemporaryDirectory() as tmp:
            output = plot_map(
                G,
                Path(tmp) / "toy.png",
                title="Toy",
                highlight_topology=True,
            )
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
