"""g5: push the planned/stale/drifted signal at agents instead of relying on
them to notice. Exp 3 showed even graph-armed agents wire into stubs blind --
get_work_context now leads with `alerts`, and propose_edge (standalone and
batch) returns a `caution` when an endpoint is not implemented.
"""

import pytest

import server
from models import Component, Edge, EdgeType, Graph
from store import add_component, add_edge


def comp(cid, z=0, parent=None, external=False, implemented=False):
    c = Component(
        component_id=cid,
        description="d",
        processing="p",
        input_types=["a"],
        output_types=["b"],
        z_level=z,
        external=external,
        parent_id=parent,
    )
    if implemented:
        c.implemented_version = c.version
    return c


@pytest.fixture
def wired_graph(monkeypatch):
    """done -> focal -> stub (planned); focal has one planned child and an
    external neighbor."""
    g = Graph.new()
    add_component(g, comp("done", implemented=True))
    add_component(g, comp("focal", implemented=True))
    add_component(g, comp("stub"))
    add_component(g, comp("ext", external=True))
    add_component(g, comp("kid", z=1, parent="focal"))
    add_edge(g, Edge(EdgeType.FLOW, "done", "focal"))
    add_edge(g, Edge(EdgeType.FLOW, "focal", "stub"))
    add_edge(g, Edge(EdgeType.FLOW, "focal", "ext"))
    monkeypatch.setattr(server, "GRAPH", g)
    monkeypatch.setattr(server, "_persist", lambda: None)
    return g


def test_work_context_leads_with_stub_alerts(wired_graph):
    out = server.get_work_context("focal")
    assert list(out.keys())[0] == "alerts"
    assert any("downstream consumer 'stub' is PLANNED" in a for a in out["alerts"])
    assert any("child 'kid' is PLANNED" in a for a in out["alerts"])
    # implemented producer and external consumer are not alerted
    assert not any("'done'" in a or "'ext'" in a for a in out["alerts"])


def test_work_context_alerts_focal_and_statuses(wired_graph):
    out = server.get_work_context("stub")
    assert any(a.startswith("THIS component 'stub' is PLANNED") for a in out["alerts"])
    stale = wired_graph.components["done"]
    stale.version += 1  # spec edited after implementation
    out = server.get_work_context("focal")
    assert any("upstream producer 'done' is STALE" in a for a in out["alerts"])


def test_work_context_no_alerts_key_when_clean(wired_graph):
    g = wired_graph
    for cid in ("stub", "kid"):
        c = g.components[cid]
        c.implemented_version = c.version
    out = server.get_work_context("focal")
    assert "alerts" not in out


def test_propose_edge_cautions_on_stub_endpoint(wired_graph):
    add_component(wired_graph, comp("newnode", implemented=True))
    out = server.propose_edge("FLOW", "newnode", "stub")
    assert out["ok"] is True
    assert any("to-endpoint 'stub' is PLANNED" in c for c in out["caution"])
    out2 = server.propose_edge("FLOW", "done", "newnode")
    assert out2["ok"] is True and "caution" not in out2


def test_batch_propose_edge_carries_caution(wired_graph):
    add_component(wired_graph, comp("other", implemented=True))
    res = server._apply_batch_op(
        wired_graph, 0, {"op": "propose_edge", "edge_type": "FLOW", "from_id": "other", "to_id": "stub"}
    )
    assert res["ok"] is True
    assert any("'stub' is PLANNED" in c for c in res["caution"])
