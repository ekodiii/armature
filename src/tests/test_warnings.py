"""Warning system: each check fires on the right shape, the flow_cycle id is
per-cycle (regression), and ignore decisions survive recomputation."""

from models import Component, Edge, EdgeType, Graph
from store import add_component, add_edge
from graph_warnings import (
    get_active_warnings,
    ignore_warning,
    run_all_warnings,
)


def comp(cid, z=0, parent=None, itypes=("t",), otypes=("t",), external=False):
    return Component(cid, "d", "p", list(itypes), list(otypes), z, external, parent_id=parent)


def types_present(graph):
    return {w.warning_type for w in run_all_warnings(graph)}


def test_hanging_output_fires():
    g = Graph.new()
    add_component(g, comp("a", otypes=["x"]))
    add_component(g, comp("b", itypes=["y"], otypes=["z"]))
    add_edge(g, Edge(EdgeType.FLOW, "a", "b"))  # b consumes y, not x
    assert "hanging_output" in types_present(g)


def test_hanging_output_clears_when_consumed():
    # `a`'s output x is consumed by b, so a must not hang. (b's own output z is
    # a separate, expected hanging_output — terminal node.)
    g = Graph.new()
    add_component(g, comp("a", otypes=["x"]))
    add_component(g, comp("b", itypes=["x"], otypes=["z"]))
    add_edge(g, Edge(EdgeType.FLOW, "a", "b"))
    hanging = {w.id for w in run_all_warnings(g) if w.warning_type == "hanging_output"}
    assert "hanging_output__a" not in hanging
    assert "hanging_output__b" in hanging


def test_boundary_output_not_hanging():
    # A child whose unconsumed output matches the parent's declared output is
    # feeding the parent boundary, not hanging. An output the parent does NOT
    # declare still hangs.
    g = Graph.new()
    add_component(g, comp("p", otypes=["x"]))
    add_component(g, comp("c1", z=1, parent="p", otypes=["x"]))
    add_component(g, comp("c2", z=1, parent="p", otypes=["y"]))
    hanging = {w.id for w in run_all_warnings(g) if w.warning_type == "hanging_output"}
    assert "hanging_output__c1" not in hanging
    assert "hanging_output__c2" in hanging


def test_root_output_still_hangs_without_consumer():
    g = Graph.new()
    add_component(g, comp("r", otypes=["z"]))
    add_component(g, comp("s", itypes=["other"], otypes=["w"]))
    add_edge(g, Edge(EdgeType.FLOW, "r", "s"))
    hanging = {w.id for w in run_all_warnings(g) if w.warning_type == "hanging_output"}
    assert "hanging_output__r" in hanging


def test_starved_input_fires_only_when_wired():
    g = Graph.new()
    add_component(g, comp("a", otypes=["x"]))
    add_component(g, comp("b", itypes=["needed"], otypes=["z"]))
    add_edge(g, Edge(EdgeType.FLOW, "a", "b"))  # a produces x, b needs `needed`
    assert "starved_input" in types_present(g)


def test_pure_entry_node_not_starved():
    # No incoming FLOW => graph boundary, not a starved input.
    g = Graph.new()
    add_component(g, comp("a", itypes=["fromoutside"], otypes=["x"]))
    add_component(g, comp("b", itypes=["x"], otypes=["z"]))
    add_edge(g, Edge(EdgeType.FLOW, "a", "b"))
    starved = [w for w in run_all_warnings(g) if w.warning_type == "starved_input"]
    assert all("a" not in w.id for w in starved)


def test_undefined_types_fires_on_empty_port():
    g = Graph.new()
    add_component(g, comp("a", itypes=[], otypes=["x"]))
    assert "undefined_types" in types_present(g)


def test_external_skips_type_warnings():
    g = Graph.new()
    add_component(g, comp("a", itypes=[], otypes=[], external=True))
    add_component(g, comp("b", otypes=["x"]))
    present = types_present(g)
    assert "undefined_types" not in present


def test_orphaned_component_fires():
    g = Graph.new()
    add_component(g, comp("a"))
    add_component(g, comp("b"))  # no edges anywhere
    assert "orphaned_component" in types_present(g)


def test_single_component_not_orphaned():
    g = Graph.new()
    add_component(g, comp("a"))
    assert "orphaned_component" not in types_present(g)


def test_flow_cycle_id_is_per_cycle():
    # Regression: the id used to be the constant "flow_cycle", so ignoring one
    # intentional loop silenced every future cycle.
    g = Graph.new()
    add_component(g, comp("a"))
    add_component(g, comp("b"))
    add_edge(g, Edge(EdgeType.FLOW, "a", "b"))
    add_edge(g, Edge(EdgeType.FLOW, "b", "a"))
    cyc = [w for w in run_all_warnings(g) if w.warning_type == "flow_cycle"]
    assert cyc and cyc[0].id != "flow_cycle"
    assert cyc[0].id.startswith("flow_cycle__")


def test_ignore_persists_across_recompute():
    g = Graph.new()
    add_component(g, comp("a", otypes=["x"]))
    add_component(g, comp("b", itypes=["y"], otypes=["z"]))
    add_edge(g, Edge(EdgeType.FLOW, "a", "b"))
    g.warnings = run_all_warnings(g)
    wid = next(w.id for w in g.warnings if w.warning_type == "hanging_output")
    ignore_warning(g.warnings, wid, "intentional log")
    # recompute (as _persist would) and confirm it stays ignored
    g.warnings = run_all_warnings(g)
    assert all(w.ignored for w in g.warnings if w.id == wid)
    assert wid not in {w.id for w in get_active_warnings(g.warnings)}
