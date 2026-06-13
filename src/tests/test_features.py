"""Feature-tag primitive: planned-feature nodes are layered beside the as-built
base and excluded from base views, so planning never pollutes the live graph.
"""

from models import Component, Edge, EdgeType, Graph
from store import add_component, add_edge
from writer import update_component
from serializer import load_graph, save_graph
from graph_warnings import run_all_warnings, get_active_warnings


def comp(cid, z=0, parent=None, itypes=("t",), otypes=("t",), feature=None):
    return Component(cid, "d", "p", list(itypes), list(otypes), z,
                     parent_id=parent, feature=feature)


def warn_ids(g):
    return {w.id for w in run_all_warnings(g)}


def test_feature_node_excluded_from_hanging_warning():
    # Both a base node and a feature node have an unconsumed output. Only the
    # base node should warn; the feature node is a proposal, not live.
    g = Graph.new()
    add_component(g, comp("base-x", otypes=["unconsumed"]))
    add_component(g, comp("feat-x", otypes=["unconsumed"], feature="F"))
    ids = warn_ids(g)
    assert "hanging_output__base-x" in ids
    assert "hanging_output__feat-x" not in ids


def test_feature_node_excluded_from_undefined_and_orphan():
    g = Graph.new()
    add_component(g, comp("real", otypes=["t"]))
    # a feature node with empty types and no edges would normally fire
    # undefined_types + orphaned_component — but it must be invisible to base.
    add_component(g, comp("feat", itypes=[], otypes=[], feature="F"))
    ids = warn_ids(g)
    assert not any(i.endswith("__feat") for i in ids)


def test_feature_cycle_not_reported_in_base():
    g = Graph.new()
    add_component(g, comp("a", feature="F"))
    add_component(g, comp("b", feature="F"))
    add_edge(g, Edge(EdgeType.FLOW, "a", "b"))
    add_edge(g, Edge(EdgeType.FLOW, "b", "a"))
    assert not any(w.warning_type == "flow_cycle" for w in run_all_warnings(g))


def test_base_cycle_still_reported_alongside_feature():
    g = Graph.new()
    add_component(g, comp("p"))
    add_component(g, comp("q"))
    add_edge(g, Edge(EdgeType.FLOW, "p", "q"))
    add_edge(g, Edge(EdgeType.FLOW, "q", "p"))
    add_component(g, comp("f", feature="F"))
    assert any(w.warning_type == "flow_cycle" for w in run_all_warnings(g))


def test_update_can_tag_and_untag():
    g = Graph.new()
    add_component(g, comp("c"))
    assert update_component(g, "c", {"feature": "F"}) == []
    assert g.components["c"].feature == "F"
    # landing: shed the tag -> rejoins base
    assert update_component(g, "c", {"feature": None}) == []
    assert g.components["c"].feature is None


def test_feature_tag_round_trips(tmp_path):
    g = Graph.new()
    add_component(g, comp("base-c"))
    add_component(g, comp("feat-c", feature="verification-router"))
    p = tmp_path / "g.yaml"
    save_graph(g, str(p))
    g2 = load_graph(str(p))
    assert g2.components["base-c"].feature is None
    assert g2.components["feat-c"].feature == "verification-router"
    # base node's feature key should not be written at all (clean base)
    import yaml
    data = yaml.safe_load(p.read_text())
    assert "feature" not in data["components"]["base-c"]
    assert data["components"]["feat-c"]["feature"] == "verification-router"
