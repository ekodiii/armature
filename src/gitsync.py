"""git-sync — drift reconciliation against git.

Out-of-band code edits (a normal editor, another agent, a teammate's commit)
change the code without going through Armature, so nothing flips a component
stale and the graph silently rots. This module closes that loop: it diffs the
repository against each component's verified baseline commit, finds the
components whose anchored line ranges overlap changed code, and marks them
code-drifted. git is the source of truth for "the code moved"; reconcile is the
act of synchronizing the graph's drift flags with that truth.

Pure-ish: the only side effect is shelling out to `git` (read-only) and setting
`code_drifted` / `last_synced_sha` on the in-memory graph. Persistence is the
caller's job. Mirrors the graph spec: diff_mapper -> mark_drift -> reconcile
(gitsync-diff-mapper / gitsync-drift-marker / gitsync-reporter).
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


# A unified-diff hunk header: @@ -old,len +new,len @@ . We only care about the
# new-file side (where the current code lives, which is what anchors point at).
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def changed_line_ranges(
    root: str,
    base: str,
    head: Optional[str] = None,
    include_uncommitted: bool = False,
) -> dict:
    """Map each changed file (repo-root-relative path) -> list of (start, end)
    line ranges that differ between `base` and `head` (default HEAD), on the
    new-file side. `--unified=0` so ranges are tight.

    include_uncommitted=True diffs `base` against the working tree instead of a
    committed ref, catching edits that are not committed yet (useful during
    active development; the default sticks to committed history for determinism).
    A pure deletion hunk (new-len 0) is recorded as a zero-width range at the
    line it was removed from, so a component anchored there still registers."""
    spec = [base] if include_uncommitted else [f"{base}..{head or 'HEAD'}"]
    diff = _git(root, "diff", "--unified=0", "--no-color", *spec)

    ranges: dict[str, list[tuple[int, int]]] = {}
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
            start = int(m.group(1))
            length = int(m.group(2)) if m.group(2) is not None else 1
            end = start + length - 1 if length > 0 else start
            ranges.setdefault(current, []).append((start, end))
    return ranges


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
    them to establish one) rather than guessed at. Advances `last_synced_sha` to
    HEAD. Pure aside from read-only git calls + in-memory flag updates; the
    caller persists the graph.
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
            changed = changed_line_ranges(root, base, head, include_uncommitted)
        except GitError:
            # an unresolvable baseline (e.g. rebased-away commit) -> can't judge
            report.no_baseline.extend(cids)
            continue
        hit = {c.component_id: c.paths for c in diff_mapper(graph, changed, prefix)}
        for cid in cids:
            comp = graph.components[cid]
            if cid in hit:
                comp.code_drifted = True
                report.drifted.append((cid, hit[cid]))
            elif comp.code_drifted:
                comp.code_drifted = False
                report.recovered.append(cid)

    graph.last_synced_sha = head
    return report
