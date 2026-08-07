"""git-sync — drift reconciliation against git.

Out-of-band code edits (a normal editor, another agent, a teammate's commit)
change the code without going through Armature, so nothing flips a component
stale and the graph silently rots. This module closes that loop: it diffs the
repository against each component's verified baseline commit, finds the
components whose anchored line ranges overlap changed code, and marks them
code-drifted. git is the source of truth for "the code moved"; reconcile is the
act of synchronizing the graph's drift flags with that truth.

Stored anchors (FileLocation.start_line/end_line) are recorded in the line
coordinates of whatever commit the component was last verified at
(implemented_sha) -- the OLD side of any later diff, not the new/HEAD side.
Two consequences drive this module's design:

1. Drift detection must intersect changed hunks against anchors using the
   OLD-file-side hunk ranges. Intersecting new-side ranges against
   baseline-coordinate anchors is a coordinate mismatch: an insertion above an
   anchor shifts its true position in the new file, and a hunk that lands
   exactly on that shifted position can miss the (differently-numbered) stored
   range entirely -- a false-clean. See `parse_hunks` / `map_line`.
2. A component whose code did NOT drift has still likely moved (anything
   inserted or deleted above it in the file). reconcile() auto-shifts such
   components' anchors to HEAD coordinates and advances their baseline --
   lossless, since the content is unchanged and only its position moved. A
   drifted component's anchors are best-effort shifted too (so drift_diff and
   get_component_code still point near the right region) but its flag and
   baseline are left for mark_implemented to clear, same as before.

Pure-ish: the only side effect is shelling out to `git` (read-only) and setting
`code_drifted` / `implemented_sha` / anchor coordinates / `last_synced_sha` on
the in-memory graph. Persistence is the caller's job. Mirrors the graph spec:
diff_mapper -> mark_drift -> reconcile (gitsync-diff-mapper /
gitsync-drift-marker / gitsync-reporter).
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from models import Graph


# ---------------------------------------------------------------------------
# git plumbing (read-only)
# ---------------------------------------------------------------------------

class GitError(RuntimeError):
    """git was unavailable, the path is not a repo, or a ref did not resolve."""


def _git(root: str, *args: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", root, *args],
            capture_output=True, text=True, check=True,
        )
    except FileNotFoundError as e:
        raise GitError("git executable not found on PATH") from e
    except subprocess.CalledProcessError as e:
        raise GitError(f"git {' '.join(args)} failed: {e.stderr.strip()}") from e
    return out.stdout


def repo_root(root: str) -> str:
    """Absolute path of the git work-tree containing `root`."""
    return _git(root, "rev-parse", "--show-toplevel").strip()


def current_head(root: str) -> str:
    """Full SHA of HEAD."""
    return _git(root, "rev-parse", "HEAD").strip()


# A unified-diff hunk header: @@ -old_start,old_len +new_start,new_len @@ . Both
# sides are captured: the old side is where stored (baseline) anchor
# coordinates live, the new side is where HEAD coordinates live. A missing
# ",len" means length 1; an explicit ",0" means a pure insertion (old side) or
# pure deletion (new side).
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True)
class Hunk:
    old_start: int
    old_n: int
    new_start: int
    new_n: int


def parse_hunks(
    root: str,
    base: str,
    head: Optional[str] = None,
    include_uncommitted: bool = False,
) -> dict:
    """Map each changed file (repo-root-relative path) -> list of `Hunk`s
    (old_start, old_n, new_start, new_n), in file order, between `base` and
    `head` (default HEAD). `--unified=0` so hunks are tight -- one per
    contiguous change, no context padding. The shared parse behind both
    `changed_line_ranges` (drift intersection) and `map_line` (anchor
    shifting), so both operate on the identical set of hunks.

    include_uncommitted=True diffs `base` against the working tree instead of
    a committed ref, catching edits that are not committed yet (useful during
    active development; the default sticks to committed history for
    determinism)."""
    spec = [base] if include_uncommitted else [f"{base}..{head or 'HEAD'}"]
    diff = _git(root, "diff", "--unified=0", "--no-color", *spec)

    hunks: dict[str, list[Hunk]] = {}
    current: Optional[str] = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            # "+++ b/path/to/file" or "+++ /dev/null" for deletions
            target = line[4:].strip()
            if target == "/dev/null":
                current = None
            else:
                current = target[2:] if target.startswith("b/") else target
        elif line.startswith("@@") and current is not None:
            m = _HUNK_RE.match(line)
            if not m:
                continue
            old_start = int(m.group(1))
            old_n = int(m.group(2)) if m.group(2) is not None else 1
            new_start = int(m.group(3))
            new_n = int(m.group(4)) if m.group(4) is not None else 1
            hunks.setdefault(current, []).append(Hunk(old_start, old_n, new_start, new_n))
    return hunks


def _ranges_from_hunks(hunks_by_path: dict, side: str) -> dict:
    """Collapse a path -> [Hunk] map into path -> [(start, end)] ranges on the
    requested side ("old" or "new"). A hunk with zero length on that side (a
    pure insertion on the old side, a pure deletion on the new side) becomes a
    zero-width range at its start line, so a component anchored exactly there
    still registers as touched."""
    ranges: dict[str, list[tuple[int, int]]] = {}
    for path, hunks in hunks_by_path.items():
        rs = []
        for h in hunks:
            s, n = (h.old_start, h.old_n) if side == "old" else (h.new_start, h.new_n)
            rs.append((s, s + n - 1) if n > 0 else (s, s))
        if rs:
            ranges[path] = rs
    return ranges


def changed_line_ranges(
    root: str,
    base: str,
    head: Optional[str] = None,
    include_uncommitted: bool = False,
    side: str = "new",
) -> dict:
    """Map each changed file (repo-root-relative path) -> list of (start, end)
    line ranges that differ between `base` and `head` (default HEAD).

    `side` selects which file's coordinates the ranges are expressed in:
    "new" (default, HEAD-side -- e.g. for diffing against current-coordinate
    anchors) or "old" (base-side -- required when intersecting against
    baseline-coordinate anchors, which is what reconcile's drift check does).
    See `parse_hunks` for the include_uncommitted contract."""
    hunks_by_path = parse_hunks(root, base, head, include_uncommitted)
    return _ranges_from_hunks(hunks_by_path, side)


def map_line(line: Optional[int], hunks: list, is_end: bool = False) -> Optional[int]:
    """Shift a baseline (old-file) line number to HEAD (new-file) coordinates
    given `hunks` (a path's `Hunk` list from `parse_hunks`, any order -- sorted
    here by old_start).

    - A line inside a replaced/deleted hunk's old range snaps to that hunk's
      new bounds (new_start for the start of a range, new_start+new_n-1 for
      the end) instead of sliding by a uniform offset, so a range whose
      boundary lands mid-edit still expands to cover the edit. A hunk that
      deletes its old lines outright (new_n == 0) collapses the line to the
      insertion point (new_start) -- there is nothing left to point at.
    - A pure-insertion hunk (old_n == 0) only pushes lines strictly AFTER
      old_start; old_start itself (the anchor line the insertion sits after)
      is untouched.
    - Lines below all of a hunk's old range accumulate its (new_n - old_n)
      offset.

    None passes through unchanged (whole-file/unanchored locations)."""
    if line is None:
        return None
    offset = 0
    for h in sorted(hunks, key=lambda h: h.old_start):
        old_end = h.old_start + h.old_n - 1
        if h.old_n > 0 and h.old_start <= line <= old_end:
            if h.new_n > 0:
                return h.new_start + h.new_n - 1 if is_end else h.new_start
            return h.new_start
        if h.old_n == 0:
            if line > h.old_start:
                offset += h.new_n
            continue
        if line > old_end:
            offset += h.new_n - h.old_n
    return line + offset


def drift_diff(
    root: str,
    base: Optional[str],
    paths: list,
    max_lines: int = 120,
) -> Optional[str]:
    """Unified `git diff <base>..HEAD -- <paths>` for a drifted component's
    anchored files -- the ground truth handed to an agent in place of (or
    alongside) a spec that may now describe code that no longer exists.

    `paths` are interpreted relative to `root` (project-root-relative anchor
    paths work directly here, unlike changed_line_ranges, since git pathspecs
    on the command line are resolved against the -C directory, not the repo
    root). Capped at `max_lines` with a trailing truncation note pointing at
    the full command. Never raises: no git, no repo, an unresolvable `base`,
    or no paths all yield None so a missing baseline degrades gracefully
    instead of breaking the read tool that calls this."""
    if not base or not paths:
        return None
    try:
        diff = _git(root, "diff", "--no-color", f"{base}..HEAD", "--", *paths)
    except GitError:
        return None
    if not diff.strip():
        return None
    lines = diff.splitlines()
    if len(lines) > max_lines:
        kept = lines[:max_lines]
        kept.append(
            f"... diff truncated; run `git diff {base}..HEAD -- {' '.join(paths)}` for the rest"
        )
        return "\n".join(kept)
    return diff


# ---------------------------------------------------------------------------
# gitsync-diff-mapper: changed hunks -> affected components
# ---------------------------------------------------------------------------

def _overlaps(loc_start: Optional[int], loc_end: Optional[int],
              hunks: list) -> bool:
    """True if a component location overlaps any changed hunk. A location with
    no line range covers the whole file, so any change to that file counts."""
    if loc_start is None and loc_end is None:
        return True
    lo = loc_start or 1
    hi = loc_end or 10**9
    for hs, he in hunks:
        if lo <= he and hi >= hs:
            return True
    return False


@dataclass
class DriftCandidate:
    component_id: str
    paths: list = field(default_factory=list)  # anchored files that changed


def diff_mapper(graph: Graph, changed: dict, path_prefix: str = "") -> list:
    """Components whose anchored line ranges overlap the changed hunks.

    `changed` must be expressed in the SAME coordinate system as the stored
    anchors -- that's the OLD (baseline) side of the diff, since anchors are
    recorded in the coordinates of the commit they were last verified at.
    Pass `changed_line_ranges(..., side="old")` here, not the default "new"
    side (which is what HEAD-coordinate readers like a live editor want).

    `path_prefix` is prepended to each component location path before matching,
    so project-root-relative anchors line up with git's repo-root-relative diff
    paths when the graph lives in a subdirectory of the repo."""
    candidates: list[DriftCandidate] = []
    for cid, comp in graph.components.items():
        if comp.external:
            continue
        hit_paths = []
        for loc in comp.locations:
            full = f"{path_prefix}{loc.path}" if path_prefix else loc.path
            full = full.replace("\\", "/")
            hunks = changed.get(full)
            if hunks and _overlaps(loc.start_line, loc.end_line, hunks):
                hit_paths.append(loc.path)
        if hit_paths:
            candidates.append(DriftCandidate(component_id=cid, paths=hit_paths))
    return candidates


# ---------------------------------------------------------------------------
# gitsync-drift-marker + gitsync-reporter, driven by reconcile()
# ---------------------------------------------------------------------------

@dataclass
class ReconcileReport:
    head: str
    base_used: Optional[str]
    drifted: list = field(default_factory=list)       # [(cid, [paths])] newly/again drifted
    recovered: list = field(default_factory=list)     # cids whose code reverted to baseline
    no_baseline: list = field(default_factory=list)   # implemented but never given a sha
    shifted: list = field(default_factory=list)        # anchors moved to HEAD coordinates
    checked: int = 0

    @property
    def is_clean(self) -> bool:
        return not self.drifted and not self.no_baseline


def _status(comp) -> str:
    if comp.implemented_version is None:
        return "planned"
    if comp.code_drifted:
        return "drifted"
    if comp.implemented_version < comp.version:
        return "stale"
    return "implemented"


def _full_path(prefix: str, path: str) -> str:
    full = f"{prefix}{path}" if prefix else path
    return full.replace("\\", "/")


def reconcile(
    graph: Graph,
    project_root: str,
    since: Optional[str] = None,
    include_uncommitted: bool = False,
) -> ReconcileReport:
    """Diff each implemented component against the commit its code was verified
    at and flag the ones whose anchored code has since changed.

    Baseline per component: its own `implemented_sha`, else `since`, else the
    graph's `last_synced_sha`. Components with no baseline are reported (re-mark
    them to establish one) rather than guessed at.

    Drift is judged on the OLD (baseline) side of the diff, matching the
    coordinate system stored anchors are in -- see the module docstring. A
    component that is NOT drifted has still likely moved (anything inserted or
    deleted above it), so its anchors are shifted to HEAD coordinates and its
    baseline (implemented_sha) is advanced to HEAD: lossless, since the
    content is unchanged and only its position moved. A DRIFTED component's
    anchors are best-effort shifted too (snap semantics -- see `map_line`) so
    downstream readers (drift_diff, get_component_code) still land near the
    right region, but its flag and baseline are left alone; only
    mark_implemented clears those. Whole-file anchors (no start_line) are
    never shifted, matching their existing any-change-is-drift behavior.

    Advances `last_synced_sha` to HEAD regardless. Pure aside from read-only
    git calls + in-memory mutations (flags, anchor coordinates, baselines);
    the caller persists the graph.
    """
    head = current_head(project_root)
    root = repo_root(project_root)
    # project_root may be a subdir of the repo; translate anchor paths (which are
    # project-root-relative) into repo-root-relative ones for diff matching.
    rel = Path(project_root).resolve().relative_to(Path(root).resolve())
    prefix = "" if str(rel) == "." else f"{rel.as_posix()}/"

    report = ReconcileReport(head=head, base_used=since)

    # Group components by the baseline they diff against, so we run one git diff
    # per distinct base rather than one per component.
    by_base: dict[str, list] = {}
    for cid, comp in graph.components.items():
        if comp.external or comp.implemented_version is None:
            continue  # only verified code can drift
        report.checked += 1
        base = comp.implemented_sha or since or graph.last_synced_sha
        if base is None:
            report.no_baseline.append(cid)
            continue
        by_base.setdefault(base, []).append(cid)

    for base, cids in by_base.items():
        if base == head and not include_uncommitted:
            # nothing committed since this baseline; clear any stale drift flags
            for cid in cids:
                if graph.components[cid].code_drifted:
                    graph.components[cid].code_drifted = False
                    report.recovered.append(cid)
            continue
        try:
            hunks_by_path = parse_hunks(root, base, head, include_uncommitted)
        except GitError:
            # an unresolvable baseline (e.g. rebased-away commit) -> can't judge
            report.no_baseline.extend(cids)
            continue
        changed_old = _ranges_from_hunks(hunks_by_path, "old")
        hit = {c.component_id: c.paths for c in diff_mapper(graph, changed_old, prefix)}
        for cid in cids:
            comp = graph.components[cid]
            drifted_now = cid in hit

            # Shift anchors toward HEAD coordinates regardless of drift status
            # -- clean anchors move losslessly, drifted ones snap best-effort.
            for loc in comp.locations:
                if loc.start_line is None or loc.end_line is None:
                    continue  # whole-file anchor: nothing to shift
                hunks = hunks_by_path.get(_full_path(prefix, loc.path))
                if not hunks:
                    continue  # this file didn't change under this base
                new_start = map_line(loc.start_line, hunks, is_end=False)
                new_end = map_line(loc.end_line, hunks, is_end=True)
                if (new_start, new_end) != (loc.start_line, loc.end_line):
                    report.shifted.append({
                        "component_id": cid,
                        "path": loc.path,
                        "old": [loc.start_line, loc.end_line],
                        "new": [new_start, new_end],
                    })
                    loc.start_line, loc.end_line = new_start, new_end

            if drifted_now:
                comp.code_drifted = True
                report.drifted.append((cid, hit[cid]))
            else:
                if comp.code_drifted:
                    comp.code_drifted = False
                    report.recovered.append(cid)
                # lossless rebaseline: content is unchanged, only position moved
                comp.implemented_sha = head

    graph.last_synced_sha = head
    return report
