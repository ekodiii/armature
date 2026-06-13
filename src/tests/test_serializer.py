"""Serialization: only authoritative state is written, derived state is rebuilt
on load, and a hand-edited file with dangling references fails clearly."""

import yaml
import pytest

from models import Component, Edge, EdgeType, Graph, Warning
from store import add_component, add_edge
from validator import validate_consistency
from serializer import load_graph, save_graph


def build_sample():
    g = Graph.new()
    add_component(g, Component("root", "d", "p", ["in"], ["out"], 0))
    add_component(g, Component("child", "d", "p", ["in"], ["out"], 1, parent_id="root"))
    add_component(g, Component("sib", "d", "p", ["out"], ["x"], 1, parent_id="root"))
    add_edge(g, Edge(EdgeType.FLOW, "child", "sib"))
    return g


def test_round_trip_preserves_components_and_edges(tmp_path):
    g = build_sample()
    p = tmp_path / "g.yaml"
    save_graph(g, str(p))
    g2 = load_graph(str(p))
    assert set(g2.components) == set(g.components)
    # SCOPE edges are derived; the FLOW edge is persisted and restored
    assert "child__sib__FLOW" in g2.edges


def test_round_trip_rebuilds_derived_fields(tmp_path):
    g = build_sample()
    p = tmp_path / "g.yaml"
    save_graph(g, str(p))
    g2 = load_graph(str(p))
    assert validate_consistency(g2) == []
    assert g2.components["root"].children == ["child", "sib"]
    assert "root__child__SCOPE" in g2.edges


def test_scope_edges_not_persisted(tmp_path):
    g = build_sample()
    p = tmp_path / "g.yaml"
    save_graph(g, str(p))
    data = yaml.safe_load(p.read_text())
    assert all(e["edge_type"] != "SCOPE" for e in data["edges"])


def test_only_ignored_warnings_persist(tmp_path):
    g = build_sample()
    g.warnings = [
        Warning(id="w1", warning_type="t", message="m", affected=[], ignored=True, ignore_reason="ok"),
        Warning(id="w2", warning_type="t", message="m", affected=[], ignored=False),
    ]
    p = tmp_path / "g.yaml"
    save_graph(g, str(p))
    data = yaml.safe_load(p.read_text())
    persisted = {w["id"] for w in data.get("warnings", [])}
    assert persisted == {"w1"}


def test_ignore_decision_reapplied_on_load(tmp_path):
    g = build_sample()
    g.warnings = [Warning(id="w1", warning_type="t", message="m", affected=[], ignored=True, ignore_reason="because")]
    p = tmp_path / "g.yaml"
    save_graph(g, str(p))
    g2 = load_graph(str(p))
    assert len(g2.warnings) == 1
    assert g2.warnings[0].ignored and g2.warnings[0].ignore_reason == "because"


def test_dangling_edge_raises_clear_error(tmp_path):
    g = build_sample()
    p = tmp_path / "g.yaml"
    save_graph(g, str(p))
    data = yaml.safe_load(p.read_text())
    data["edges"].append({"edge_type": "FLOW", "from_id": "root", "to_id": "ghost"})
    p.write_text(yaml.dump(data, sort_keys=False))
    with pytest.raises(ValueError, match="does not exist"):
        load_graph(str(p))


def test_dangling_parent_raises_clear_error(tmp_path):
    g = build_sample()
    p = tmp_path / "g.yaml"
    save_graph(g, str(p))
    data = yaml.safe_load(p.read_text())
    data["components"]["child"]["parent_id"] = "ghost"
    p.write_text(yaml.dump(data, sort_keys=False))
    with pytest.raises(ValueError, match="parent"):
        load_graph(str(p))
