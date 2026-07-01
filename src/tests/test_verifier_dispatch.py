"""verify_contract MCP tool (vr-dispatch-verify) — the operator-facing surface.

Exercises the tool end to end against a real active graph + anchored leaf: an
operator-authored property that holds returns 'verified'; a false one returns
'refuted' with a counterexample; guards reject non-leaf components and empty
property lists.
"""

import server
from models import Component, FileLocation, Graph
from store import add_component


def _graph_with_leaf(tmp_path, body="def double(x: int):\n    return x + x\n"):
    (tmp_path / "leaf.py").write_text(body)
    g = Graph.new()
    add_component(g, Component(
        "double", "doubles a number", "double x", ["int"], ["int"], 0,
        locations=[FileLocation("leaf.py", 1, 2)],
    ))
    return g


def _activate(monkeypatch, tmp_path, graph):
    monkeypatch.setattr(server, "GRAPH", graph)
    monkeypatch.setattr(server, "PROJECT_ROOT_OVERRIDE", str(tmp_path))


def test_verify_contract_verified(tmp_path, monkeypatch):
    _activate(monkeypatch, tmp_path, _graph_with_leaf(tmp_path))
    out = server.verify_contract(
        "double", [{"name": "doubles", "expression": "result == 2 * x"}], timeout_s=60,
    )
    assert out["ok"] is True
    assert out["status"] == "verified", out
    assert "def test_double" in out["evidence"]


def test_verify_contract_refuted(tmp_path, monkeypatch):
    _activate(monkeypatch, tmp_path, _graph_with_leaf(tmp_path))
    out = server.verify_contract(
        "double", [{"name": "small", "expression": "result < 100"}], timeout_s=60,
    )
    assert out["status"] == "refuted"
    assert out["counterexample"] and "x=" in out["counterexample"]


def test_verify_contract_untyped_param_via_strategy(tmp_path, monkeypatch):
    _activate(monkeypatch, tmp_path, _graph_with_leaf(tmp_path, "def double(x):\n    return x + x\n"))
    out = server.verify_contract(
        "double", [{"name": "doubles", "expression": "result == 2 * x"}],
        strategies={"x": "st.integers()"}, timeout_s=60,
    )
    assert out["status"] == "verified", out


def test_verify_contract_rejects_non_leaf(tmp_path, monkeypatch):
    g = _graph_with_leaf(tmp_path)
    g.components["double"].children.append("some-child")
    _activate(monkeypatch, tmp_path, g)
    out = server.verify_contract("double", [{"name": "p", "expression": "True"}])
    assert "error" in out and "leaf" in out["error"]


def test_verify_contract_requires_properties(tmp_path, monkeypatch):
    _activate(monkeypatch, tmp_path, _graph_with_leaf(tmp_path))
    out = server.verify_contract("double", [])
    assert "error" in out
