import re
import sys
from pathlib import Path

# Armature root on sys.path so we can import models, etc.
_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models import Component, Edge, EdgeType, FileLocation  # noqa: E402

from .models import GraphDraft, RawNode, StructureMap  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(node_id: str) -> str:
    """Convert node_id to a valid component_id.

    path/to/file.py::Class::method → path_to_file_py__Class__method
    The "::" scope separator is preserved as "__"; other special chars become "_".
    """
    parts = node_id.split("::")
    cleaned = []
    for part in parts:
        p = re.sub(r"[^a-zA-Z0-9_]", "_", part)
        p = re.sub(r"_+", "_", p).strip("_")
        cleaned.append(p)
    return "__".join(cleaned)


def _z_level(node: RawNode) -> int:
    """module=0, class or top-level function=1, method inside class=2."""
    depth = len(node.node_id.split("::")) - 1
    return min(depth, 2)


def _parent_node_id(node: RawNode) -> str | None:
    """Return the node_id of this node's parent, or None for module-level nodes."""
    parts = node.node_id.split("::")
    if len(parts) <= 1:
        return None
    return "::".join(parts[:-1])


# ---------------------------------------------------------------------------
# gs_component_builder
# ---------------------------------------------------------------------------

def gs_component_builder(
    annotated_structure: StructureMap,
) -> tuple[list[Component], dict[str, str]]:
    """Map each annotated node to an Armature Component. Returns (components, node_id→slug)."""
    slug_map = {n.node_id: _slugify(n.node_id) for n in annotated_structure.nodes}
    components: list[Component] = []

    for node in annotated_structure.nodes:
        slug = slug_map[node.node_id]
        parent_nid = _parent_node_id(node)
        parent_slug = slug_map.get(parent_nid) if parent_nid else None

        locations: list[FileLocation] = []
        if node.relative_path:
            locations = [FileLocation(
                path=node.relative_path,
                start_line=node.start_line if node.start_line > 0 else None,
                end_line=node.end_line if node.end_line > 0 else None,
            )]

        components.append(Component(
            component_id=slug,
            description=node.processing or node.node_id.split("::")[-1],
            processing=node.processing or f"Implementation of {node.node_id.split('::')[-1]}.",
            input_types=node.input_types or [],
            output_types=node.output_types or [],
            z_level=_z_level(node),
            external=node.external,
            parent_id=parent_slug,
            locations=locations,
        ))

    return components, slug_map


# ---------------------------------------------------------------------------
# gs_edge_resolver
# ---------------------------------------------------------------------------

def gs_edge_resolver(
    components: list[Component],
    slug_map: dict[str, str],
    annotated_structure: StructureMap,
) -> tuple[list[Component], list[Edge]]:
    """Derive FLOW (same parent) and REFERENCE (cross-parent) edges."""
    slug_to_comp = {c.component_id: c for c in components}
    edges: list[Edge] = []
    seen: set[str] = set()

    def _add(from_slug: str, to_slug: str) -> None:
        if from_slug == to_slug:
            return
        if from_slug not in slug_to_comp or to_slug not in slug_to_comp:
            return
        fc = slug_to_comp[from_slug]
        tc = slug_to_comp[to_slug]
        # External stubs only get REFERENCE; same parent → FLOW; cross → REFERENCE
        if tc.external or fc.parent_id != tc.parent_id:
            etype = EdgeType.REFERENCE
        else:
            etype = EdgeType.FLOW
        edge = Edge(edge_type=etype, from_id=from_slug, to_id=to_slug)
        if edge.edge_id not in seen:
            seen.add(edge.edge_id)
            edges.append(edge)

    all_pairs = annotated_structure.call_edges + [
        (f, t) for f, t in annotated_structure.import_edges
        if not t.startswith("external:")
    ]
    for from_id, to_id in all_pairs:
        fs = slug_map.get(from_id)
        ts = slug_map.get(to_id)
        if fs and ts:
            _add(fs, ts)

    return components, edges


# ---------------------------------------------------------------------------
# gs_hierarchy_validator
# ---------------------------------------------------------------------------

def gs_hierarchy_validator(
    components: list[Component],
    edges: list[Edge],
) -> tuple[list[Component], list[Edge], list[str]]:
    """Pre-flight: parent-chain DAG, z_level consistency, FLOW sibling rule."""
    slug_to_comp = {c.component_id: c for c in components}
    issues: list[str] = []

    for comp in components:
        visited: set[str] = set()
        cur: str | None = comp.component_id
        while cur:
            if cur in visited:
                issues.append(f"Cycle in parent chain at: {comp.component_id}")
                break
            visited.add(cur)
            c = slug_to_comp.get(cur)
            cur = c.parent_id if c else None

    for comp in components:
        if comp.parent_id:
            parent = slug_to_comp.get(comp.parent_id)
            if parent and comp.z_level != parent.z_level + 1:
                issues.append(
                    f"z_level mismatch: {comp.component_id} (z={comp.z_level}) "
                    f"under {comp.parent_id} (z={parent.z_level})"
                )

    for edge in edges:
        if edge.edge_type == EdgeType.FLOW:
            fc = slug_to_comp.get(edge.from_id)
            tc = slug_to_comp.get(edge.to_id)
            if fc and tc and fc.parent_id != tc.parent_id:
                issues.append(f"FLOW crosses parents: {edge.from_id} → {edge.to_id}")

    return components, edges, issues


# ---------------------------------------------------------------------------
# gs_draft_assembler
# ---------------------------------------------------------------------------

def gs_draft_assembler(
    components: list[Component],
    edges: list[Edge],
    validation_issues: list[str],
    source_root: str,
) -> GraphDraft:
    """Package everything into a GraphDraft ready for graph_commit."""
    return GraphDraft(
        components=components,
        edges=edges,
        validation_issues=validation_issues,
        source_root=source_root,
        stats={
            "total": len(components),
            "external_count": sum(1 for c in components if c.external),
            "edge_count": len(edges),
        },
    )


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def synthesize(annotated_structure: StructureMap, source_root: str) -> GraphDraft:
    components, slug_map = gs_component_builder(annotated_structure)
    components, edges = gs_edge_resolver(components, slug_map, annotated_structure)
    components, edges, issues = gs_hierarchy_validator(components, edges)
    return gs_draft_assembler(components, edges, issues, source_root)
