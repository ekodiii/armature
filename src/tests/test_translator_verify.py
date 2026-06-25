"""Verifier: trust sub-scores stay in [0, 1], the grounding score no longer
double-counts unverifiable findings (regression), and anchor/coverage findings
fire on fiction and missed code."""

import pytest

from models import Component, Edge, EdgeType, FileLocation, Graph
from store import add_component, add_edge
from translator.lift import _filter_skeleton_to_scope, prepare_lift
from translator.skeleton import build_skeleton
from translator.source_ingestion import ingest
from translator.verify import (
    GroundingFinding,
    VerificationState,
    discrepancy_reconciler,
    verify,
)


def make_tree(tmp_path, files):
    for rel, text in files.items():
        fp = tmp_path / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(text)
    return str(tmp_path)


def anchored(cid, path, itypes=("t",), otypes=("t",), z=0, parent=None):
    return Component(cid, "d", "p", list(itypes), list(otypes), z, parent_id=parent,
                     locations=[FileLocation(path)])


# --------------------------------------------------------------------------- #
# Grounding score (regression for the double-count / negative score bug)
# --------------------------------------------------------------------------- #

def test_grounding_score_no_double_count(tmp_path):
    # 5 FLOW edges; 3 unverifiable + 2 ungrounded findings.
    # Correct: verifiable = 5 - 3 = 2, ungrounded = 2 -> 1 - 2/2 = 0.0
    # Old bug:  1 - 5/2 = -1.5
    g = Graph.new()
    for i in range(10):
        add_component(g, Component(f"c{i}", "d", "p", ["t"], ["t"], 0))
    for i in range(0, 10, 2):
        add_edge(g, Edge(EdgeType.FLOW, f"c{i}", f"c{i+1}"))
    skel = build_skeleton(ingest(make_tree(tmp_path, {"x.py": "def f():\n    return 1\n"})),
                          str(tmp_path))
    st = VerificationState(graph=g, skeleton=skel)
    st.grounding_findings = (
        [GroundingFinding("e", "a", "b", "FLOW", "x", kind="unverifiable")] * 3
        + [GroundingFinding("e", "a", "b", "FLOW", "x", kind="ungrounded")] * 2
    )
    vm = discrepancy_reconciler(st)
    assert vm.trust_breakdown["grounding_score"] == 0.0


def test_all_subscores_bounded(tmp_path):
    root = make_tree(tmp_path, {
        "a.py": "def produce():\n    return 1\n\ndef consume(x):\n    return x\n",
    })
    skel = build_skeleton(ingest(root), root)
    g = Graph.new()
    add_component(g, anchored("p", "a.py", otypes=["v"]))
    add_component(g, anchored("c", "a.py", itypes=["v"]))
    add_edge(g, Edge(EdgeType.FLOW, "p", "c"))
    vm = verify(g, skel)
    for k, v in vm.trust_breakdown.items():
        if k.endswith("_score"):
            assert 0.0 <= v <= 1.0, (k, v)
    assert 0.0 <= vm.trust_score <= 1.0


# --------------------------------------------------------------------------- #
# Anchor + coverage findings
# --------------------------------------------------------------------------- #

def test_anchor_finding_for_fictional_file(tmp_path):
    root = make_tree(tmp_path, {"a.py": "def f():\n    return 1\n"})
    skel = build_skeleton(ingest(root), root)
    g = Graph.new()
    add_component(g, anchored("ghost", "does_not_exist.py"))
    vm = verify(g, skel)
    reasons = {f.reason for f in vm.anchor_findings}
    assert "file_not_in_skeleton" in reasons


def test_anchorless_node_flagged(tmp_path):
    root = make_tree(tmp_path, {"a.py": "def f():\n    return 1\n"})
    skel = build_skeleton(ingest(root), root)
    g = Graph.new()
    add_component(g, Component("nowhere", "d", "p", ["t"], ["t"], 0))  # no locations
    vm = verify(g, skel)
    kinds = {f.kind for f in vm.coverage_findings}
    assert "anchorless_node" in kinds


def test_uncovered_symbol_flagged(tmp_path):
    # Two functions in the file; the graph covers neither -> uncovered findings.
    root = make_tree(tmp_path, {"a.py": "def f():\n    return 1\n\ndef g():\n    return 2\n"})
    skel = build_skeleton(ingest(root), root)
    graph = Graph.new()  # empty graph
    vm = verify(graph, skel)
    uncovered = {f.subject for f in vm.coverage_findings if f.kind == "uncovered_symbol"}
    assert any("a.py::f" in s for s in uncovered)
    assert any("a.py::g" in s for s in uncovered)


def test_external_component_skips_anchor_checks(tmp_path):
    root = make_tree(tmp_path, {"a.py": "def f():\n    return 1\n"})
    skel = build_skeleton(ingest(root), root)
    g = Graph.new()
    ext = Component("db", "d", "p", ["t"], ["t"], 0, external=True)
    add_component(g, ext)
    vm = verify(g, skel)
    assert all(f.component_id != "db" for f in vm.anchor_findings)
    assert all(f.subject != "db" for f in vm.coverage_findings)


def test_clean_graph_high_trust(tmp_path):
    # A graph that covers every leaf symbol with a real anchor scores well.
    root = make_tree(tmp_path, {"a.py": "def only():\n    return 1\n"})
    skel = build_skeleton(ingest(root), root)
    g = Graph.new()
    add_component(g, anchored("c", "a.py"))
    vm = verify(g, skel)
    assert vm.trust_breakdown["coverage_score"] == 1.0
    assert vm.trust_breakdown["anchor_score"] == 1.0


# --------------------------------------------------------------------------- #
# Hardening: precision (coarse inclusion-only blobs) + externals (boundaries)
# --------------------------------------------------------------------------- #

def test_coarse_leaf_flagged(tmp_path):
    # A leaf anchored to many whole files is an inclusion-only blob.
    files = {f"m{i}.py": "def f():\n    return 1\n" for i in range(6)}
    root = make_tree(tmp_path, files)
    skel = build_skeleton(ingest(root), root)
    g = Graph.new()
    add_component(g, Component(
        "blob", "d", "p", ["t"], ["t"], 0,
        locations=[FileLocation(f"m{i}.py") for i in range(6)],
    ))
    vm = verify(g, skel)
    assert any(f.component_id == "blob" for f in vm.precision_findings)
    assert vm.trust_breakdown["precision_score"] < 1.0


def test_focused_leaf_not_flagged(tmp_path):
    # A focused leaf (one file, few symbols) must NOT be flagged coarse —
    # this is the hand-authored pattern; hardening must not punish it.
    root = make_tree(tmp_path, {"a.py": "def f():\n    return 1\n"})
    skel = build_skeleton(ingest(root), root)
    g = Graph.new()
    add_component(g, anchored("c", "a.py"))
    vm = verify(g, skel)
    assert vm.precision_findings == []
    assert vm.trust_breakdown["precision_score"] == 1.0


def test_missing_external_flagged(tmp_path):
    # Third-party import + ZERO external nodes -> missing-boundary finding.
    root = make_tree(tmp_path, {"a.py": "import zzz_thirdparty\n\ndef f():\n    return 1\n"})
    skel = build_skeleton(ingest(root), root)
    g = Graph.new()
    add_component(g, anchored("c", "a.py"))
    vm = verify(g, skel)
    assert any(f.module == "zzz_thirdparty" for f in vm.external_findings)
    assert vm.trust_breakdown["externals_score"] == 0.0


def test_external_node_suppresses_external_finding(tmp_path):
    # If the graph DOES model a boundary node, don't fire (conservative check).
    root = make_tree(tmp_path, {"a.py": "import zzz_thirdparty\n\ndef f():\n    return 1\n"})
    skel = build_skeleton(ingest(root), root)
    g = Graph.new()
    add_component(g, anchored("c", "a.py"))
    add_component(g, Component("boundary", "d", "p", ["t"], ["t"], 0, external=True))
    vm = verify(g, skel)
    assert vm.external_findings == []
    assert vm.trust_breakdown["externals_score"] == 1.0


# --------------------------------------------------------------------------- #
# Scope filtering: out-of-scope symbols must NOT appear as uncovered
# --------------------------------------------------------------------------- #

def _make_pkg_tests_tree(tmp_path):
    """Two directories: pkg/ (in-scope) and tests/ (out-of-scope)."""
    files = {
        "pkg/__init__.py": "",
        "pkg/core.py": "def do_thing():\n    return 1\n\ndef helper():\n    return 2\n",
        "tests/__init__.py": "",
        "tests/test_core.py": (
            "from pkg.core import do_thing\n\n"
            "def test_do_thing():\n    assert do_thing() == 1\n"
        ),
    }
    for rel, text in files.items():
        fp = tmp_path / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(text)
    return str(tmp_path)


def test_scope_filter_excludes_out_of_scope_symbols(tmp_path):
    """_filter_skeleton_to_scope with scope='pkg/' must not include tests/ symbols."""
    root = _make_pkg_tests_tree(tmp_path)
    full_skel = build_skeleton(ingest(root), root)
    filtered = _filter_skeleton_to_scope(full_skel, "pkg/")

    full_paths = {s.path for s in full_skel.symbols}
    filtered_paths = {s.path for s in filtered.symbols}

    # Full skeleton sees both trees
    assert any(p.startswith("tests/") for p in full_paths)
    assert any(p.startswith("pkg/") for p in full_paths)

    # Filtered skeleton sees only pkg/
    assert not any(p.startswith("tests/") for p in filtered_paths), (
        "tests/ symbols leaked into scope-filtered skeleton"
    )
    assert any(p.startswith("pkg/") for p in filtered_paths)


def test_scope_filter_none_returns_all_symbols(tmp_path):
    """scope=None must keep the full skeleton unchanged."""
    root = _make_pkg_tests_tree(tmp_path)
    full_skel = build_skeleton(ingest(root), root)
    filtered = _filter_skeleton_to_scope(full_skel, None)
    assert filtered is full_skel  # must be the same object


def test_scoped_verify_does_not_penalise_tests_dir(tmp_path):
    """translate_verify scoped to pkg/ must not report tests/ symbols as uncovered."""
    root = _make_pkg_tests_tree(tmp_path)
    # Build a skeleton scoped to pkg/ only (the fix path)
    full_skel = build_skeleton(ingest(root), root)
    scoped_skel = _filter_skeleton_to_scope(full_skel, "pkg/")

    # Graph covers all of pkg/ with one component
    g = Graph.new()
    add_component(g, Component(
        "pkg_core", "d", "p", ["t"], ["t"], 0,
        locations=[FileLocation("pkg/core.py")],
    ))
    vm = verify(g, scoped_skel)

    uncovered_subjects = {f.subject for f in vm.coverage_findings if f.kind == "uncovered_symbol"}
    tests_uncovered = [s for s in uncovered_subjects if s.startswith("tests/")]
    assert tests_uncovered == [], (
        f"tests/ symbols reported as uncovered when scope='pkg/': {tests_uncovered}"
    )


def test_unscoped_verify_includes_tests_dir(tmp_path):
    """Without scope (scope=None), tests/ symbols ARE counted as uncovered when
    the graph only covers pkg/ — confirming scope=None behavior is unchanged."""
    root = _make_pkg_tests_tree(tmp_path)
    full_skel = build_skeleton(ingest(root), root)

    # Graph covers only pkg/
    g = Graph.new()
    add_component(g, Component(
        "pkg_core", "d", "p", ["t"], ["t"], 0,
        locations=[FileLocation("pkg/core.py")],
    ))
    vm = verify(g, full_skel)

    uncovered_subjects = {f.subject for f in vm.coverage_findings if f.kind == "uncovered_symbol"}
    tests_uncovered = [s for s in uncovered_subjects if s.startswith("tests/")]
    assert len(tests_uncovered) > 0, (
        "Expected tests/ symbols to be uncovered when scope=None and graph only covers pkg/"
    )


def test_prepare_lift_scope_skeleton_excludes_out_of_scope(tmp_path):
    """prepare_lift(root, scope='pkg/') must cache a skeleton that excludes tests/ symbols
    so that coverage_tracker / verify report correctly."""
    root = _make_pkg_tests_tree(tmp_path)
    ctx = prepare_lift(root, scope="pkg/")

    skel_paths = {s.path for s in ctx.skeleton.symbols}
    assert not any(p.startswith("tests/") for p in skel_paths), (
        "prepare_lift cached skeleton includes out-of-scope tests/ symbols"
    )
