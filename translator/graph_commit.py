import sys
from collections import defaultdict, deque
from pathlib import Path

_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from graph_warnings import run_all_warnings  # noqa: E402
from models import Graph  # noqa: E402
from serializer import save_graph  # noqa: E402
from writer import propose_component, propose_edge  # noqa: E402

from .models import GraphDraft, MigrationReport, WriteLog  # noqa: E402


# ---------------------------------------------------------------------------
# gc_topological_sorter
# ---------------------------------------------------------------------------

def gc_topological_sorter(draft: GraphDraft) -> GraphDraft:
    """Sort components parents-before-children; orphans (missing parent) go last."""
    comp_map = {c.component_id: c for c in draft.components}
    in_degree: dict[str, int] = {cid: 0 for cid in comp_map}
    children_of: dict[str, list[str]] = defaultdict(list)
    issues: list[str] = list(draft.issues)

    for comp in draft.components:
        if comp.parent_id:
            if comp.parent_id in comp_map:
                in_degree[comp.component_id] += 1
                children_of[comp.parent_id].append(comp.component_id)
            else:
                issues.append(
                    f"Orphan: {comp.component_id} — parent {comp.parent_id} not in draft"
                )
                comp.parent_id = None  # write without parent as fallback

    queue: deque[str] = deque(
        sorted(cid for cid, d in in_degree.items() if d == 0)
    )
    ordered = []
    while queue:
        cid = queue.popleft()
        ordered.append(comp_map[cid])
        for child_id in sorted(children_of[cid]):
            in_degree[child_id] -= 1
            if in_degree[child_id] == 0:
                queue.append(child_id)

    # Handle any cycles left over
    remaining = {c.component_id for c in draft.components} - {c.component_id for c in ordered}
    for cid in sorted(remaining):
        ordered.append(comp_map[cid])
        issues.append(f"Cycle or unresolvable hierarchy: {cid}")

    return GraphDraft(
        components=ordered,
        edges=draft.edges,
        validation_issues=draft.validation_issues,
        source_root=draft.source_root,
        stats=draft.stats,
        issues=issues,
    )


# ---------------------------------------------------------------------------
# gc_component_writer
# ---------------------------------------------------------------------------

def gc_component_writer(sorted_draft: GraphDraft, graph: Graph) -> WriteLog:
    """Write each component via propose_component; accumulate errors, never abort."""
    log = WriteLog()

    for comp in sorted_draft.components:
        errors = propose_component(graph, comp)
        if errors:
            log.failed.append({"component_id": comp.component_id, "errors": errors})
        else:
            log.written.append(comp.component_id)

    return log


# ---------------------------------------------------------------------------
# gc_edge_writer
# ---------------------------------------------------------------------------

def gc_edge_writer(
    sorted_draft: GraphDraft,
    write_log: WriteLog,
    graph: Graph,
) -> WriteLog:
    """Write each edge; skip if either endpoint failed to write."""
    written_set = set(write_log.written)
    failed_set = {entry["component_id"] for entry in write_log.failed}

    for edge in sorted_draft.edges:
        from_failed = edge.from_id in failed_set
        to_failed = edge.to_id in failed_set
        from_missing = edge.from_id not in written_set
        to_missing = edge.to_id not in written_set

        if from_failed or to_failed or from_missing or to_missing:
            reason = "endpoint write failed" if (from_failed or to_failed) else "endpoint not written"
            write_log.edge_skipped.append({"edge_id": edge.edge_id, "reason": reason})
            continue

        errors = propose_edge(graph, edge)
        if errors:
            write_log.edge_failed.append({"edge_id": edge.edge_id, "errors": errors})
        else:
            write_log.edge_written.append(edge.edge_id)

    return write_log


# ---------------------------------------------------------------------------
# gc_error_collector
# ---------------------------------------------------------------------------

def gc_error_collector(write_log: WriteLog, sorted_draft: GraphDraft) -> MigrationReport:
    """Summarise all write failures and validation issues."""
    summary = (
        f"{len(write_log.written)} components written, "
        f"{len(write_log.failed)} failed, "
        f"{len(write_log.edge_written)} edges written, "
        f"{len(write_log.edge_skipped)} skipped"
    )
    all_issues = sorted_draft.validation_issues + sorted_draft.issues
    if write_log.failed or write_log.edge_failed:
        print(f"[armature-translator] {summary}")
        for entry in write_log.failed:
            print(f"  FAILED component {entry['component_id']}: {entry['errors']}")

    return MigrationReport(
        summary=summary,
        failures=write_log.failed,
        skipped_edges=write_log.edge_skipped,
        validation_issues=all_issues,
        active_warnings=[],
    )


# ---------------------------------------------------------------------------
# gc_persist_trigger
# ---------------------------------------------------------------------------

def gc_persist_trigger(
    report: MigrationReport,
    graph: Graph,
    path: str,
) -> MigrationReport:
    """Recompute warnings, flush to YAML, append warning count to report."""
    graph.warnings = run_all_warnings(graph)
    save_graph(graph, path)
    active = [w.id for w in graph.warnings if not w.ignored]
    report.active_warnings = active
    report.summary += f"; {len(active)} active warnings after persist"
    return report


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def commit(draft: GraphDraft, graph: Graph, path: str) -> MigrationReport:
    sorted_draft = gc_topological_sorter(draft)
    write_log = gc_component_writer(sorted_draft, graph)
    write_log = gc_edge_writer(sorted_draft, write_log, graph)
    report = gc_error_collector(write_log, sorted_draft)
    return gc_persist_trigger(report, graph, path)
