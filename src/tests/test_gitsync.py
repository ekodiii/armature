"""git-sync drift reconciliation, exercised against real temp git repos.

Covers the full loop the graph spec describes: a component verified at one
commit, code edited under its anchor, reconcile flags it 'drifted'; re-marking
recovers it; edits outside its line range don't flag it; a missing baseline is
reported rather than guessed.
"""

import subprocess
import textwrap

import pytest

from models import Component, FileLocation, Graph
from store import add_component
from writer import mark_implemented
from serializer import load_graph, save_graph
from gitsync import GitError, changed_line_ranges, diff_mapper, reconcile, current_head


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True,
                   capture_output=True, text=True)


def init_repo(root):
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@t.t")
    git(root, "config", "user.name", "t")
    git(root, "config", "commit.gpgsign", "false")


def commit_all(root, msg):
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", msg)
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def write(root, name, text):
    (root / name).write_text(textwrap.dedent(text))


def comp(cid, path, start=None, end=None):
    return Component(cid, "d", "p", ["t"], ["t"], 0,
                     locations=[FileLocation(path, start, end)])


# --------------------------------------------------------------------------- #
# diff parsing
# --------------------------------------------------------------------------- #

def test_changed_line_ranges_detects_edited_lines(tmp_path):
    init_repo(tmp_path)
    write(tmp_path, "a.py", "def f():\n    return 1\n\n\ndef g():\n    return 2\n")
    base = commit_all(tmp_path, "init")
    write(tmp_path, "a.py", "def f():\n    return 999\n\n\ndef g():\n    return 2\n")
    head = commit_all(tmp_path, "edit f")
    changed = changed_line_ranges(str(tmp_path), base, head)
    assert "a.py" in changed
    # the edited line (2) is within the reported ranges
    assert any(lo <= 2 <= hi for lo, hi in changed["a.py"])


# --------------------------------------------------------------------------- #
# end-to-end reconcile
# --------------------------------------------------------------------------- #

def build_repo_and_graph(tmp_path):
    init_repo(tmp_path)
    write(tmp_path, "a.py", "def f():\n    return 1\n\n\ndef g():\n    return 2\n")
    base = commit_all(tmp_path, "init")
    g = Graph.new()
    add_component(g, comp("comp-f", "a.py", 1, 2))   # anchors def f
    add_component(g, comp("comp-g", "a.py", 5, 6))   # anchors def g
    for cid in ("comp-f", "comp-g"):
        mark_implemented(g, cid, sha=base)
    return g, base


def test_reconcile_flags_only_the_changed_component(tmp_path):
    g, base = build_repo_and_graph(tmp_path)
    # edit only f's body
    write(tmp_path, "a.py", "def f():\n    return 999\n\n\ndef g():\n    return 2\n")
    commit_all(tmp_path, "edit f")

    report = reconcile(g, str(tmp_path))
    drifted = {cid for cid, _ in report.drifted}
    assert drifted == {"comp-f"}
    assert g.components["comp-f"].code_drifted is True
    assert g.components["comp-g"].code_drifted is False
    assert not report.is_clean


def test_remark_recovers_drifted_component(tmp_path):
    g, base = build_repo_and_graph(tmp_path)
    write(tmp_path, "a.py", "def f():\n    return 999\n\n\ndef g():\n    return 2\n")
    head = commit_all(tmp_path, "edit f")

    reconcile(g, str(tmp_path))
    assert g.components["comp-f"].code_drifted is True

    # operator re-verifies f against the new commit
    mark_implemented(g, "comp-f", sha=head)
    assert g.components["comp-f"].code_drifted is False
    report = reconcile(g, str(tmp_path))
    assert g.components["comp-f"].code_drifted is False
    assert "comp-f" not in {cid for cid, _ in report.drifted}


def test_reconcile_clean_when_nothing_changed(tmp_path):
    g, base = build_repo_and_graph(tmp_path)
    report = reconcile(g, str(tmp_path))
    assert report.is_clean
    assert report.drifted == []
    assert g.last_synced_sha == current_head(str(tmp_path))


def test_no_baseline_reported_not_guessed(tmp_path):
    g, base = build_repo_and_graph(tmp_path)
    # a component implemented without a sha (e.g. authored before git-sync)
    add_component(g, comp("comp-nobase", "a.py", 1, 2))
    mark_implemented(g, "comp-nobase", sha=None)
    g.last_synced_sha = None  # and no graph-level fallback
    report = reconcile(g, str(tmp_path))
    assert "comp-nobase" in report.no_baseline


def test_planned_and_external_components_skipped(tmp_path):
    g, base = build_repo_and_graph(tmp_path)
    add_component(g, comp("comp-planned", "a.py", 1, 2))  # never marked
    ext = Component("ext", "d", "p", ["t"], ["t"], 0, external=True)
    add_component(g, ext)
    write(tmp_path, "a.py", "def f():\n    return 999\n\n\ndef g():\n    return 2\n")
    commit_all(tmp_path, "edit")
    report = reconcile(g, str(tmp_path))
    drifted = {cid for cid, _ in report.drifted}
    assert "comp-planned" not in drifted and "ext" not in drifted


def test_subdir_graph_paths_translated(tmp_path):
    # graph anchors are project-root-relative; the repo root is one level up.
    init_repo(tmp_path)
    (tmp_path / "pkg").mkdir()
    write(tmp_path, "pkg/a.py", "def f():\n    return 1\n")
    base = commit_all(tmp_path, "init")
    g = Graph.new()
    add_component(g, comp("c", "a.py", 1, 2))  # path relative to pkg/, not repo root
    mark_implemented(g, "c", sha=base)
    write(tmp_path, "pkg/a.py", "def f():\n    return 2\n")
    commit_all(tmp_path, "edit")
    report = reconcile(g, str(tmp_path / "pkg"))  # project_root = pkg/
    assert "c" in {cid for cid, _ in report.drifted}


def test_reconcile_raises_cleanly_outside_repo(tmp_path):
    g = Graph.new()
    add_component(g, comp("c", "a.py", 1, 2))
    mark_implemented(g, "c", sha="deadbeef")
    with pytest.raises(GitError):
        reconcile(g, str(tmp_path))  # tmp_path is not a git repo


# --------------------------------------------------------------------------- #
# status + serialization of the new fields
# --------------------------------------------------------------------------- #

def test_serialization_round_trips_drift_fields(tmp_path):
    g, base = build_repo_and_graph(tmp_path)
    g.components["comp-f"].code_drifted = True
    g.last_synced_sha = base
    p = tmp_path / "g.yaml"
    save_graph(g, str(p))
    g2 = load_graph(str(p))
    assert g2.components["comp-f"].code_drifted is True
    assert g2.components["comp-f"].implemented_sha == base
    assert g2.components["comp-g"].code_drifted is False
    assert g2.last_synced_sha == base
