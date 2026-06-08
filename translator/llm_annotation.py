"""LLM annotation layer.

The LLM executing the pipeline IS the annotator — no external API call is needed.
`la_batch_annotator` accepts an `annotate_fn` callback with the signature:

    annotate_fn(context: dict) -> dict
        context keys: node_id, kind, params, return_hint, snippet, callee_types, external_flag
        return keys:  input_types (list[str]), output_types (list[str]), processing (str)

When driven by a Claude agent, the agent supplies this callback by reasoning inline
over the context dict. A heuristic fallback is provided for headless/testing runs.
"""

from collections import defaultdict, deque
from typing import Callable, Optional

from .models import FileRecord, RawNode, StructureMap


# ---------------------------------------------------------------------------
# la_traversal_orderer — Kahn's algorithm, leaves first
# ---------------------------------------------------------------------------

def la_traversal_orderer(structure_map: StructureMap) -> list[RawNode]:
    """Topologically sort nodes so callees come before callers (leaves first)."""
    node_ids = {n.node_id for n in structure_map.nodes}
    id_to_node = {n.node_id: n for n in structure_map.nodes}

    out_degree: dict[str, int] = {nid: 0 for nid in node_ids}
    callers_of: dict[str, list[str]] = defaultdict(list)

    for from_id, to_id in structure_map.call_edges:
        if from_id in node_ids and to_id in node_ids:
            out_degree[from_id] += 1
            callers_of[to_id].append(from_id)

    queue: deque[str] = deque(
        sorted(nid for nid, d in out_degree.items() if d == 0)
    )
    depth: dict[str, int] = {nid: 0 for nid in node_ids}
    ordered: list[RawNode] = []
    visited: set[str] = set()

    while queue:
        nid = queue.popleft()
        if nid in visited:
            continue
        visited.add(nid)
        node = id_to_node[nid]
        node.depth = depth[nid]
        ordered.append(node)
        for caller_id in callers_of[nid]:
            depth[caller_id] = max(depth[caller_id], depth[nid] + 1)
            out_degree[caller_id] -= 1
            if out_degree[caller_id] == 0:
                queue.append(caller_id)

    # Cycle survivors — deterministic order, depth stays 0 (CYCLE_NOTE)
    for nid in sorted(node_ids - visited):
        node = id_to_node[nid]
        node.depth = 0
        ordered.append(node)

    return ordered


# ---------------------------------------------------------------------------
# lba_context_builder
# ---------------------------------------------------------------------------

def lba_context_builder(
    node: RawNode,
    annotation_cache: dict[str, dict],
    file_contents: dict[str, str],
) -> dict:
    """Assemble the self-contained annotation context for one node."""
    snippet = ""
    if node.relative_path and node.relative_path in file_contents:
        lines = file_contents[node.relative_path].splitlines()
        if node.start_line > 0:
            start = max(0, node.start_line - 1)
            end = min(len(lines), node.end_line if node.end_line > 0 else node.start_line + 60)
            snippet = "\n".join(lines[start:end])[:3000]

    callee_types: dict[str, list[str]] = {}
    for call_id in node.calls:
        if call_id in annotation_cache:
            callee_types[call_id] = annotation_cache[call_id].get("output_types", [])

    return {
        "node_id": node.node_id,
        "kind": node.kind,
        "params": node.params,
        "return_hint": node.return_hint,
        "snippet": snippet,
        "callee_types": callee_types,
        "external_flag": node.external,
    }


# ---------------------------------------------------------------------------
# lba_type_inferrer  (heuristic fallback — real LLM reasoning goes in annotate_fn)
# ---------------------------------------------------------------------------

def lba_type_inferrer(context: dict, annotate_fn: Callable[[dict], dict]) -> dict:
    """Apply annotate_fn to the context; return TypedContext."""
    if context["external_flag"]:
        name = context["node_id"].split("::")[-1]
        return {
            **context,
            "input_types": [],
            "output_types": ["ExternalResult"],
            "processing": f"Atomic external boundary — {name}.",
            "external": True,
        }
    result = annotate_fn(context)
    return {**context, **result, "external": context["external_flag"]}


# ---------------------------------------------------------------------------
# lba_description_writer
# ---------------------------------------------------------------------------

def lba_description_writer(typed_context: dict) -> dict:
    """Pass-through: processing description is expected from annotate_fn / lba_type_inferrer."""
    return typed_context


# ---------------------------------------------------------------------------
# la_batch_annotator
# ---------------------------------------------------------------------------

def la_batch_annotator(
    ordered_nodes: list[RawNode],
    file_contents: dict[str, str],
    annotate_fn: Callable[[dict], dict],
) -> list[RawNode]:
    """Core annotation loop: process nodes bottom-up, cache callee types for each step."""
    annotation_cache: dict[str, dict] = {}

    for node in ordered_nodes:
        context = lba_context_builder(node, annotation_cache, file_contents)
        typed = lba_type_inferrer(context, annotate_fn)
        final = lba_description_writer(typed)

        node.input_types = final.get("input_types") or []
        node.output_types = final.get("output_types") or []
        node.processing = final.get("processing", "")
        node.external = final.get("external", node.external)

        annotation_cache[node.node_id] = {
            "input_types": node.input_types,
            "output_types": node.output_types,
        }

    return ordered_nodes


# ---------------------------------------------------------------------------
# la_annotation_merger
# ---------------------------------------------------------------------------

def la_annotation_merger(
    annotated_nodes: list[RawNode],
    structure_map: StructureMap,
) -> StructureMap:
    """Merge per-node annotations back into the original StructureMap topology."""
    id_to_annotated = {n.node_id: n for n in annotated_nodes}

    merged: list[RawNode] = []
    for node in structure_map.nodes:
        an = id_to_annotated.get(node.node_id, node)
        if not an.external and not an.input_types and not an.output_types:
            # Slipped through unannotated; graph_commit will surface the warning
            an.input_types = ["undefined_types"]
            an.output_types = ["undefined_types"]
        merged.append(an)

    return StructureMap(
        nodes=merged,
        call_edges=structure_map.call_edges,
        import_edges=structure_map.import_edges,
    )


# ---------------------------------------------------------------------------
# Heuristic fallback annotate_fn (no LLM required)
# ---------------------------------------------------------------------------

def _heuristic_annotate(context: dict) -> dict:
    """Minimal rule-based annotation for testing / headless runs."""
    kind = context["kind"]
    params = context["params"]
    return_hint = context["return_hint"]
    callee_types = context["callee_types"]

    # Prefer callee output type as our input type if exactly one callee
    callee_outputs = [v for vals in callee_types.values() for v in vals]
    input_types = callee_outputs[:1] if len(callee_types) == 1 else []

    if kind == "module":
        return {"input_types": [], "output_types": ["Module"], "processing": "Module-level namespace."}

    if return_hint:
        output_types = [return_hint.split("[")[0].strip()]
    elif callee_outputs:
        output_types = [callee_outputs[-1]]
    else:
        output_types = ["Result"]

    name = context["node_id"].split("::")[-1]
    processing = (
        f"Receives {input_types[0] if input_types else (params[0] if params else 'input')}, "
        f"emits {output_types[0]}."
    )
    return {"input_types": input_types or ["Input"], "output_types": output_types, "processing": processing}


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def annotate(
    structure_map: StructureMap,
    file_manifest: list[FileRecord],
    annotate_fn: Optional[Callable[[dict], dict]] = None,
) -> StructureMap:
    """Annotate all nodes. Pass annotate_fn=None to use the heuristic fallback."""
    file_contents = {f.relative_path: f.text for f in file_manifest}
    fn = annotate_fn if annotate_fn is not None else _heuristic_annotate
    ordered = la_traversal_orderer(structure_map)
    annotated = la_batch_annotator(ordered, file_contents, fn)
    return la_annotation_merger(annotated, structure_map)
