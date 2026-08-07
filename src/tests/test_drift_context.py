"""Spec demotion + diff-on-drift: get_work_context must not hand an agent a
stale spec as if it were still the acceptance criterion. A controlled
experiment showed the plain 'THIS component is DRIFTED' alert gets overridden
-- agents force the code back to match the stale spec, deleting whatever now
occupies its anchored lines. The fix demotes the spec and surfaces the actual
diff since the verified baseline as the thing to adapt to (or stop over).

Mimics test_gitsync.py's temp-git-repo fixtures; exercised through
server.get_work_context (and gitsync.drift_diff directly for the cap/no-git
edge cases) rather than through the MCP tool wrapper.
"""

import subprocess
import textwrap

import pytest

import server
from models import Component, Edge, EdgeType, FileLocation, Graph
from store import add_component, add_edge
from writer import mark_implemented
from gitsync import reconcile, drift_diff, GitError


# --------------------------------------------------------------------------- #
# helpers (same shape as test_gitsync.py)
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


def comp(cid, path, start=None, end=None, itypes=("t",), otypes=("t",)):
    return Component(cid, "d", "spec text naming the old symbol", list(itypes),
                     list(otypes), 0, locations=[FileLocation(path, start, end)])


def build_repo_and_graph(tmp_path):
    """comp-f anchors def f (lines 1-2), comp-g anchors def g (lines 5-6);
    comp-f flows into comp-g. Both marked implemented at `base`."""
    init_repo(tmp_path)
    write(tmp_path, "a.py", "def f():\n    return 1\n\n\ndef g():\n    return 2\n")
    base = commit_all(tmp_path, "init")
    g = Graph.new()
    add_component(g, comp("comp-f", "a.py", 1, 2))
    add_component(g, comp("comp-g", "a.py", 5, 6))
    add_edge(g, Edge(EdgeType.FLOW, "comp-f", "comp-g"))
    for cid in ("comp-f", "comp-g"):
        mark_implemented(g, cid, sha=base)
    return g, base


def activate(monkeypatch, g, root):
    monkeypatch.setattr(server, "GRAPH", g)
    monkeypatch.setattr(server, "PROJECT_ROOT_OVERRIDE", str(root))
    monkeypatch.setattr(server, "_persist", lambda: None)


# --------------------------------------------------------------------------- #
# (a) focal drifted: drift key, real diff content, demoted processing
# --------------------------------------------------------------------------- #

def test_focal_drift_surfaces_diff_and_demotes_spec(tmp_path, monkeypatch):
    g, base = build_repo_and_graph(tmp_path)
    write(tmp_path, "a.py", "def f():\n    return 999\n\n\ndef g():\n    return 2\n")
    commit_all(tmp_path, "edit f")
    reconcile(g, str(tmp_path))
    assert g.components["comp-f"].code_drifted is True

    activate(monkeypatch, g, tmp_path)
    out = server.get_work_context("comp-f")

    assert "drift" in out
    assert out["drift"]["baseline"] == base
    assert "999" in out["drift"]["diff"]  # the actual changed line, not a summary
    assert "-    return 1" in out["drift"]["diff"] or "return 1" in out["drift"]["diff"]
    note = out["drift"]["note"]
    assert "UNVERIFIED" in note
    assert "DIFF is the truth" in note
    assert "Do NOT restore" in note

    # demotion: processing is prefixed, not silently swapped or mutated in the graph
    assert out["component"]["processing"].startswith("[UNVERIFIED")
    assert "drift.diff" in out["component"]["processing"]
    assert "spec text naming the old symbol" in out["component"]["processing"]
    assert g.components["comp-f"].processing == "spec text naming the old symbol"  # not mutated


# --------------------------------------------------------------------------- #
# (b) connected (dependency) drift: alert line carries a diff snippet
# --------------------------------------------------------------------------- #

def test_drifted_dependency_alert_carries_diff_snippet(tmp_path, monkeypatch):
    g, base = build_repo_and_graph(tmp_path)
    write(tmp_path, "a.py", "def f():\n    return 999\n\n\ndef g():\n    return 2\n")
    commit_all(tmp_path, "edit f")
    reconcile(g, str(tmp_path))
    assert g.components["comp-f"].code_drifted is True
    assert g.components["comp-g"].code_drifted is False

    activate(monkeypatch, g, tmp_path)
    out = server.get_work_context("comp-g")

    assert "drift" not in out  # comp-g itself is not drifted
    upstream_alerts = [a for a in out["alerts"] if "comp-f" in a]
    assert len(upstream_alerts) == 1
    alert = upstream_alerts[0]
    assert "DRIFTED" in alert
    assert "999" in alert  # diff snippet inline
    assert "Do NOT restore" in alert


# --------------------------------------------------------------------------- #
# (c) diff cap + truncation note
# --------------------------------------------------------------------------- #

def test_drift_diff_caps_with_truncation_note(tmp_path):
    init_repo(tmp_path)
    write(tmp_path, "a.py", "".join(f"line{i}\n" for i in range(1, 21)))
    base = commit_all(tmp_path, "init")
    write(tmp_path, "a.py", "".join(f"changed{i}\n" for i in range(1, 21)))
    commit_all(tmp_path, "rewrite all lines")

    diff = drift_diff(str(tmp_path), base, ["a.py"], max_lines=5)
    assert diff is not None
    lines = diff.splitlines()
    assert len(lines) == 6  # 5 kept + truncation note
    assert "truncated" in lines[-1]
    assert f"git diff {base}..HEAD -- a.py" in lines[-1]


# --------------------------------------------------------------------------- #
# (d) no git / no baseline: graceful, no crash, no drift key
# --------------------------------------------------------------------------- #

def test_drift_diff_no_baseline_returns_none(tmp_path):
    init_repo(tmp_path)
    write(tmp_path, "a.py", "x = 1\n")
    commit_all(tmp_path, "init")
    assert drift_diff(str(tmp_path), None, ["a.py"]) is None


def test_drift_diff_not_a_repo_returns_none(tmp_path):
    write(tmp_path, "a.py", "x = 1\n")
    assert drift_diff(str(tmp_path), "deadbeef", ["a.py"]) is None


def test_work_context_no_baseline_no_crash_no_drift_key(tmp_path, monkeypatch):
    g, base = build_repo_and_graph(tmp_path)
    # simulate a component flagged drifted with no recorded baseline sha
    # (e.g. authored before git-sync existed)
    g.components["comp-f"].code_drifted = True
    g.components["comp-f"].implemented_sha = None

    activate(monkeypatch, g, tmp_path)
    out = server.get_work_context("comp-f")  # must not raise

    assert "drift" not in out
    assert not out["component"]["processing"].startswith("[UNVERIFIED")


# --------------------------------------------------------------------------- #
# (e) clean component: no drift key, no prefix
# --------------------------------------------------------------------------- #

def test_clean_component_has_no_drift_key_or_prefix(tmp_path, monkeypatch):
    g, base = build_repo_and_graph(tmp_path)
    activate(monkeypatch, g, tmp_path)
    out = server.get_work_context("comp-f")

    assert "drift" not in out
    assert out["component"]["processing"] == "spec text naming the old symbol"
