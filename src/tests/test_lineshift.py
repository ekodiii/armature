"""git line-shift auto-tracking: the offset mapper (`parse_hunks` / `map_line`)
and reconcile()'s auto-shift-on-clean / best-effort-shift-on-drift behavior.

Anchors are stored in the line coordinates of whatever commit a component was
last verified at (implemented_sha) -- the OLD side of any later diff. Two
things this file locks down:

1. Drift detection must intersect changed hunks against anchors using
   OLD-file-side ranges, not NEW-file-side ranges. Using the new side is a
   coordinate mismatch that produces false-cleans when something is inserted
   above an anchor and the anchor's *real* edit happens to land where the
   insertion hunk's NEW-side numbering doesn't overlap the anchor's stale OLD
   coordinates. `test_new_side_ranges_miss_the_real_edit_old_side_catches_it`
   demonstrates the mismatch directly; `test_edit_inside_shifted_anchor_still_
   drifts_despite_insertions_above` is the same thing exercised through
   reconcile().
2. A component that is NOT drifted has still likely moved (anything inserted
   or deleted above it). reconcile() auto-shifts such anchors to HEAD
   coordinates and advances implemented_sha -- lossless, since content didn't
   change. A drifted component's anchors are best-effort shifted too (snap
   semantics) but its flag and baseline are left alone.
"""

import subprocess
import textwrap

from models import Component, FileLocation, Graph
from store import add_component
from writer import mark_implemented
from gitsync import (
    Hunk,
    changed_line_ranges,
    map_line,
    parse_hunks,
    reconcile,
)


# --------------------------------------------------------------------------- #
# helpers (mirrors tests/test_gitsync.py)
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


ORIGINAL = "def f():\n    return 1\n\n\ndef g():\n    return 2\n"


def build_repo_and_graph(tmp_path):
    """f anchored 1-2, g anchored 5-6, both verified at the initial commit."""
    init_repo(tmp_path)
    write(tmp_path, "a.py", ORIGINAL)
    base = commit_all(tmp_path, "init")
    g = Graph.new()
    add_component(g, comp("comp-f", "a.py", 1, 2))
    add_component(g, comp("comp-g", "a.py", 5, 6))
    for cid in ("comp-f", "comp-g"):
        mark_implemented(g, cid, sha=base)
    return g, base


# --------------------------------------------------------------------------- #
# (a) insertions above an anchor -> not drifted, shifted, baseline advanced
# --------------------------------------------------------------------------- #

def test_insertion_above_anchor_shifts_without_drift(tmp_path):
    g, base = build_repo_and_graph(tmp_path)
    write(tmp_path, "a.py", "# c1\n# c2\n# c3\n" + ORIGINAL)
    head = commit_all(tmp_path, "insert header comment")

    report = reconcile(g, str(tmp_path))

    assert report.drifted == []
    gc = g.components["comp-g"]
    assert gc.code_drifted is False
    assert (gc.locations[0].start_line, gc.locations[0].end_line) == (8, 9)
    assert gc.implemented_sha == head  # lossless rebaseline

    fc = g.components["comp-f"]
    assert (fc.locations[0].start_line, fc.locations[0].end_line) == (4, 5)
    assert fc.implemented_sha == head

    shifted_ids = {s["component_id"] for s in report.shifted}
    assert shifted_ids == {"comp-f", "comp-g"}
    g_entry = next(s for s in report.shifted if s["component_id"] == "comp-g")
    assert g_entry["old"] == [5, 6]
    assert g_entry["new"] == [8, 9]


# --------------------------------------------------------------------------- #
# (b) the old/new-side mismatch regression
# --------------------------------------------------------------------------- #

def test_new_side_ranges_miss_the_real_edit_old_side_catches_it(tmp_path):
    """Direct proof the bug was real: insert 3 lines above comp-g AND edit
    comp-g's body in the same commit. The NEW-side hunk range for the edit
    does not overlap comp-g's STALE (baseline/old-coordinate) stored range,
    so intersecting new-side ranges against baseline anchors is a false-clean.
    The OLD-side range for that same edit does overlap -- because old-side
    coordinates are what the stored anchor is actually expressed in."""
    g, base = build_repo_and_graph(tmp_path)
    write(tmp_path, "a.py", "# c1\n# c2\n# c3\n" + ORIGINAL.replace("return 2", "return 999"))
    head = commit_all(tmp_path, "insert header + edit g")

    new_side = changed_line_ranges(str(tmp_path), base, head, side="new")
    old_side = changed_line_ranges(str(tmp_path), base, head, side="old")

    gc_start, gc_end = 5, 6  # comp-g's stored (stale) coordinates
    new_overlaps = any(lo <= gc_end and gc_start <= hi for lo, hi in new_side["a.py"])
    old_overlaps = any(lo <= gc_end and gc_start <= hi for lo, hi in old_side["a.py"])

    assert new_overlaps is False, "new-side ranges should NOT line up with stale old-coordinate anchors"
    assert old_overlaps is True, "old-side ranges must catch the edit against baseline-coordinate anchors"


def test_edit_inside_shifted_anchor_still_drifts_despite_insertions_above(tmp_path):
    g, base = build_repo_and_graph(tmp_path)
    write(tmp_path, "a.py", "# c1\n# c2\n# c3\n" + ORIGINAL.replace("return 2", "return 999"))
    commit_all(tmp_path, "insert header + edit g")

    report = reconcile(g, str(tmp_path))

    drifted_ids = {cid for cid, _ in report.drifted}
    assert "comp-g" in drifted_ids
    assert g.components["comp-g"].code_drifted is True
    # f wasn't touched (besides the shift) -> stays clean
    assert g.components["comp-f"].code_drifted is False


# --------------------------------------------------------------------------- #
# (c) snap semantics: a replaced hunk overlapping the anchor's start
# --------------------------------------------------------------------------- #

def test_replaced_hunk_overlapping_start_snaps_and_expands(tmp_path):
    g, base = build_repo_and_graph(tmp_path)
    # replace old lines 4-5 (blank line + "def g():") with "X" / "async def
    # g():" -- a same-length replace hunk (old_start=4, old_n=2) whose old
    # range overlaps comp-g's anchor start (5) but not its end (6).
    new_text = "def f():\n    return 1\n\nX\nasync def g():\n    return 2\n"
    write(tmp_path, "a.py", new_text)
    commit_all(tmp_path, "expand before g")

    report = reconcile(g, str(tmp_path))

    gc = g.components["comp-g"]
    assert gc.code_drifted is True  # the hunk does touch g's old range (line 5)
    start, end = gc.locations[0].start_line, gc.locations[0].end_line
    # start snaps to the hunk's new_start (4) instead of a naive +0 uniform
    # shift (which would leave it at 5); end (6) is outside the hunk's old
    # range so it keeps its normal offset (+0 here, same-length replace).
    # Net: the anchor widens from (5,6) to (4,6) to cover the whole edit.
    assert (start, end) == (4, 6)


# --------------------------------------------------------------------------- #
# (d) deletions above -> negative shift
# --------------------------------------------------------------------------- #

def test_deletion_above_anchor_shifts_negative(tmp_path):
    g, base = build_repo_and_graph(tmp_path)
    # delete "def f():" and "    return 1" (old lines 1-2)
    new_text = "\n\ndef g():\n    return 2\n"
    write(tmp_path, "a.py", new_text)
    commit_all(tmp_path, "delete f")

    report = reconcile(g, str(tmp_path))

    gc = g.components["comp-g"]
    assert gc.code_drifted is False
    assert (gc.locations[0].start_line, gc.locations[0].end_line) == (3, 4)


# --------------------------------------------------------------------------- #
# (e) two components, different baselines, same file, shifted independently
# --------------------------------------------------------------------------- #

def test_components_with_different_baselines_shift_independently(tmp_path):
    g, base1 = build_repo_and_graph(tmp_path)
    # commit2: insert 2 lines above everything
    write(tmp_path, "a.py", "# x\n# y\n" + ORIGINAL)
    base2 = commit_all(tmp_path, "insert 2 lines")

    # simulate comp-f having been independently re-verified at base2, already
    # at base2 coordinates -- while comp-g is left on its original baseline.
    g.components["comp-f"].locations[0].start_line = 3
    g.components["comp-f"].locations[0].end_line = 4
    mark_implemented(g, "comp-f", sha=base2)
    assert g.components["comp-g"].implemented_sha == base1

    # commit3: insert 1 more line above everything
    write(tmp_path, "a.py", "# z\n# x\n# y\n" + ORIGINAL)
    commit_all(tmp_path, "insert 1 more line")

    report = reconcile(g, str(tmp_path))

    # comp-f only saw the 1-line insertion from its own base2 baseline
    fc = g.components["comp-f"]
    assert (fc.locations[0].start_line, fc.locations[0].end_line) == (4, 5)
    # comp-g saw both insertions (3 lines total) from its base1 baseline
    gc = g.components["comp-g"]
    assert (gc.locations[0].start_line, gc.locations[0].end_line) == (8, 9)
    assert report.drifted == []


# --------------------------------------------------------------------------- #
# (f) drifted component's anchors best-effort shifted, flag + baseline held
# --------------------------------------------------------------------------- #

def test_drifted_component_anchors_shifted_but_flag_and_baseline_held(tmp_path):
    g, base = build_repo_and_graph(tmp_path)
    write(tmp_path, "a.py", "# c1\n# c2\n# c3\n" + ORIGINAL.replace("return 2", "return 999"))
    commit_all(tmp_path, "insert header + edit g")

    report = reconcile(g, str(tmp_path))

    gc = g.components["comp-g"]
    assert gc.code_drifted is True
    # best-effort shifted despite being drifted
    assert (gc.locations[0].start_line, gc.locations[0].end_line) == (8, 9)
    # baseline NOT advanced -- only mark_implemented clears drift
    assert gc.implemented_sha == base
    shifted_ids = {s["component_id"] for s in report.shifted}
    assert "comp-g" in shifted_ids


# --------------------------------------------------------------------------- #
# (g) whole-file anchors: unchanged shifting behavior
# --------------------------------------------------------------------------- #

def test_whole_file_anchor_unaffected_by_shifting(tmp_path):
    init_repo(tmp_path)
    write(tmp_path, "a.py", ORIGINAL)
    base = commit_all(tmp_path, "init")
    g = Graph.new()
    add_component(g, comp("comp-whole", "a.py"))  # no start/end -> whole file
    mark_implemented(g, "comp-whole", sha=base)

    write(tmp_path, "a.py", "# c1\n# c2\n# c3\n" + ORIGINAL)
    commit_all(tmp_path, "insert header")

    report = reconcile(g, str(tmp_path))

    wc = g.components["comp-whole"]
    assert wc.code_drifted is True  # any change to the file drifts a whole-file anchor
    assert wc.locations[0].start_line is None
    assert wc.locations[0].end_line is None
    assert report.shifted == []  # nothing to shift for an unanchored location


# --------------------------------------------------------------------------- #
# unit-level coverage of parse_hunks / map_line
# --------------------------------------------------------------------------- #

def test_parse_hunks_captures_both_sides(tmp_path):
    init_repo(tmp_path)
    write(tmp_path, "a.py", ORIGINAL)
    base = commit_all(tmp_path, "init")
    write(tmp_path, "a.py", "# c1\n# c2\n# c3\n" + ORIGINAL.replace("return 2", "return 999"))
    head = commit_all(tmp_path, "insert + edit")

    hunks = parse_hunks(str(tmp_path), base, head)["a.py"]
    assert len(hunks) == 2
    insert, edit = sorted(hunks, key=lambda h: h.old_start)
    assert insert.old_n == 0  # pure insertion
    assert insert.new_n == 3
    assert edit.old_n == 1 and edit.new_n == 1


def test_map_line_pure_insertion_only_shifts_lines_strictly_after():
    hunks = [Hunk(old_start=2, old_n=0, new_start=3, new_n=3)]
    assert map_line(2, hunks) == 2   # the anchor line itself: untouched
    assert map_line(3, hunks) == 6   # strictly after: shifted by 3
    assert map_line(1, hunks) == 1   # before: untouched


def test_map_line_negative_offset_from_deletion():
    hunks = [Hunk(old_start=1, old_n=2, new_start=1, new_n=0)]
    assert map_line(5, hunks) == 3
    assert map_line(5, hunks, is_end=True) == 3


def test_map_line_none_passthrough():
    assert map_line(None, [Hunk(1, 0, 1, 5)]) is None
