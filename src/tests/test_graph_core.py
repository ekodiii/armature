"""Core graph engine: store wiring, validation rules, writes, and cycles.

Several of these are regression tests for specific bugs:
  - component id validation (empty / "__" delimiter collision)
  - root z_level can't be changed via update_component
  - detect_cycles returns the real cycle path, ignores non-cycle nodes, and is
    iterative (no RecursionError on long chains)
"""

import sys

import pytest

from models import Component, Edge, EdgeType, Graph
from store import add_component, add_edge
from validator import detect_cycles, validate_consistency
from writer import propose_component, propose_edge, update_component, delete_component


def comp(cid, z=0, parent=None, itypes=("t",), otypes=("t",), external=False):
    return Component(
        component_id=cid,
        description="d",
        processing="p",
        input_types=list(itypes),
        output_types=list(otypes),
        z_level=z,
        external=external,
        parent_id=parent,
    )


def flow_graph(*edges):
    """Build a graph from FLOW edges given as (from, to) tuples (z=0 siblings)."""
    g = Graph.new()
    nodes = {n for e in edges for n in e}
    for n in sorted(nodes):
        add_component(g, comp(n))
    for a, b in edges:
        add_edge(g, Edge(EdgeType.FLOW, a, b))
    return g


# --------------------------------------------------------------------------- #
# Component id validation
# --------------------------------------------------------------------------- #

def test_propose_rejects_empty_id():
    g = Graph.new()
    errs = propose_component(g, comp(""))
    assert errs and "non-empty" in errs[0]


def test_propose_rejects_blank_id():
    g = Graph.new()
    assert propose_component(g, comp("   "))


def test_propose_rejects_double_underscore_id():
    # "__" is the edge-id ("a__b__FLOW") and warning-id delimiter, so an id
    # containing it would make those ambiguous.
    g = Graph.new()
    errs = propose_component(g, comp("a__b"))
    assert errs and "may not contain" in errs[0]


def test_propose_accepts_normal_id_and_wires_it():
    g = Graph.new()
    assert propose_component(g, comp("root")) == []
    assert "root" in g.components


def test_propose_rejects_duplicate_id():
    g = Graph.new()
    propose_component(g, comp("root"))
    assert propose_component(g, comp("root"))


# --------------------------------------------------------------------------- #
# Component structural validation
# --------------------------------------------------------------------------- #

def test_root_must_be_z0():
    g = Graph.new()
    assert propose_component(g, comp("r", z=3))


def test_child_z_must_be_parent_plus_one():
    g = Graph.new()
    propose_component(g, comp("p", z=0))
    assert propose_component(g, comp("c", z=2, parent="p"))
    assert propose_component(g, comp("c", z=1, parent="p")) == []


def test_child_of_missing_parent_rejected():
    g = Graph.new()
    assert propose_component(g, comp("c", z=1, parent="ghost"))


def test_external_cannot_take_children():
    g = Graph.new()
    propose_component(g, comp("ext", external=True))
    assert propose_component(g, comp("c", z=1, parent="ext"))


def test_proposing_child_materializes_scope_edge():
    g = Graph.new()
    propose_component(g, comp("p"))
    propose_component(g, comp("c", z=1, parent="p"))
    assert "p__c__SCOPE" in g.edges
    assert "c" in g.components["p"].children


# --------------------------------------------------------------------------- #
# update_component
# --------------------------------------------------------------------------- #

def test_update_rejects_protected_fields():
    g = Graph.new()
    propose_component(g, comp("root"))
    assert update_component(g, "root", {"parent_id": "x"})


def test_update_bumps_version():
    g = Graph.new()
    propose_component(g, comp("root"))
    v0 = g.components["root"].version
    update_component(g, "root", {"description": "new"})
    assert g.components["root"].version == v0 + 1


def test_update_cannot_move_root_off_z0():
    # Regression: a root has no incident SCOPE edge, so the edge-revalidation
    # path could not catch an illegal z-level change.
    g = Graph.new()
    propose_component(g, comp("root"))
    errs = update_component(g, "root", {"z_level": 5})
    assert errs and "z_level 0" in errs[0]
    assert g.components["root"].z_level == 0


def test_update_z_level_revalidates_scope_edges():
    g = Graph.new()
    propose_component(g, comp("p"))
    propose_component(g, comp("c", z=1, parent="p"))
    # moving the child to z=3 breaks the SCOPE edge contract (parent+1)
    assert update_component(g, "c", {"z_level": 3})
    assert g.components["c"].z_level == 1  # rolled back


def test_mark_external_with_children_rejected():
    g = Graph.new()
    propose_component(g, comp("p"))
    propose_component(g, comp("c", z=1, parent="p"))
    assert update_component(g, "p", {"external": True})


# --------------------------------------------------------------------------- #
# Edges
# --------------------------------------------------------------------------- #

def test_flow_between_non_siblings_rejected():
    g = Graph.new()
    propose_component(g, comp("a"))
    propose_component(g, comp("p"))
    propose_component(g, comp("b", z=1, parent="p"))
    assert propose_edge(g, Edge(EdgeType.FLOW, "a", "b"))


def test_flow_self_loop_rejected():
    g = Graph.new()
    propose_component(g, comp("a"))
    assert propose_edge(g, Edge(EdgeType.FLOW, "a", "a"))


def test_reference_may_cross_levels():
    g = Graph.new()
    propose_component(g, comp("a"))
    propose_component(g, comp("p"))
    propose_component(g, comp("b", z=1, parent="p"))
    assert propose_edge(g, Edge(EdgeType.REFERENCE, "a", "b")) == []


def test_duplicate_edge_rejected():
    g = flow_graph(("a", "b"))
    assert propose_edge(g, Edge(EdgeType.FLOW, "a", "b"))


# --------------------------------------------------------------------------- #
# delete_component
# --------------------------------------------------------------------------- #

def test_delete_removes_incident_edges():
    g = flow_graph(("a", "b"), ("b", "c"))
    delete_component(g, "b")
    assert "b" not in g.components
    assert not any("b" in eid.split("__") for eid in g.edges)
    assert validate_consistency(g) == []


# --------------------------------------------------------------------------- #
# Cycle detection (regression: path, isolation, recursion)
# --------------------------------------------------------------------------- #

def test_detect_cycle_returns_members_only():
    g = flow_graph(("A", "B"), ("B", "C"), ("C", "A"), ("D", "A"))
    cyc = detect_cycles(g, EdgeType.FLOW)
    assert set(cyc) == {"A", "B", "C"}
    assert "D" not in cyc


def test_no_cycle_returns_empty():
    g = flow_graph(("X", "Y"), ("Y", "Z"))
    assert detect_cycles(g, EdgeType.FLOW) == []


def test_cycle_detection_does_not_leak_state_across_roots():
    # Two separate components in the graph; the acyclic one must not inherit
    # stale path state from a previous DFS root.
    g = flow_graph(("A", "B"), ("B", "A"), ("P", "Q"))
    cyc = detect_cycles(g, EdgeType.FLOW)
    assert set(cyc) == {"A", "B"}


def test_cycle_detection_handles_deep_chain_without_recursion():
    g = Graph.new()
    n = 4000
    for i in range(n):
        add_component(g, comp(f"n{i}"))
    for i in range(n - 1):
        add_edge(g, Edge(EdgeType.FLOW, f"n{i}", f"n{i + 1}"))
    # Would RecursionError under a recursive DFS.
    assert detect_cycles(g, EdgeType.FLOW) == []
