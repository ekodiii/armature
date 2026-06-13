from typing import Optional

from models import Component, Edge, Graph
from store import add_component, add_edge, get_component, get_edge, remove_component
from validator import validate_component, validate_edge


# Fields a caller may set through update_component. Structural fields
# (ids, edges, children, parent_id, version) are managed by the store and
# editing them directly would desync the graph.
EDITABLE_FIELDS = {
    "description",
    "processing",
    "input_types",
    "output_types",
    "external",
    "locations",
    "z_level",
}


# Component ids are interpolated into edge ids ("from__to__TYPE") and warning
# ids ("type__component_id"), so "__" in an id would make those ambiguous (an
# edge a__b -> c could not be told from a -> b__c, and the per-component warning
# filter would match the wrong node). Reject it, along with empty/blank ids.
def _validate_component_id(component_id: str) -> list[str]:
    if not component_id or not component_id.strip():
        return ["Component id must be a non-empty, non-blank string"]
    if "__" in component_id:
        return [f"Component id '{component_id}' may not contain '__' (reserved as the edge/warning id delimiter)"]
    return []


# Component Functions #
def propose_component(graph: Graph, component: Component) -> list[str]:
    id_errors = _validate_component_id(component.component_id)
    if id_errors:
        return id_errors
    if component.component_id in graph.components:
        return [f"Component '{component.component_id}' already exists"]
    errors = validate_component(graph, component)
    if errors:
        return errors
    add_component(graph, component)
    return []


def update_component(graph: Graph, component_id: str, fields: dict) -> list[str]:
    component = get_component(graph, component_id)

    protected = [k for k in fields if k not in EDITABLE_FIELDS]
    if protected:
        return [f"Cannot update protected fields: {sorted(protected)}"]

    if fields.get("external") and component.children:
        return [f"Cannot mark '{component_id}' external: it has children"]

    new_z = fields.get("z_level", component.z_level)
    if new_z != component.z_level:
        # A root (no parent) must stay at z_level 0 — it has no incident SCOPE
        # edge to catch the violation in the edge re-validation below, so guard
        # it explicitly (mirrors the propose-time rule in validate_component).
        if component.parent_id is None and new_z != 0:
            return [f"Root component '{component_id}' must stay at z_level 0, got {new_z}"]
        # Tentatively apply the new z-level and re-validate every incident edge
        # (FLOW must stay level, SCOPE must stay one apart), then roll back.
        old_z = component.z_level
        component.z_level = new_z
        errors = [
            err
            for edge_id in component.edges_in + component.edges_out
            if (err := validate_edge(graph, get_edge(graph, edge_id))) is not None
        ]
        if errors:
            component.z_level = old_z
            return errors

    for key, value in fields.items():
        setattr(component, key, value)
    component.version += 1
    return []


def delete_component(graph: Graph, component_id: str) -> None:
    remove_component(graph, get_component(graph, component_id))


def mark_implemented(graph: Graph, component_id: str, sha: Optional[str] = None) -> list[str]:
    """Record that the component's code has been written against its current spec
    version. Does not bump version (the spec did not change); it advances the
    implemented marker to match. After any later update_component the version moves
    ahead again and the component reads as stale.

    `sha` is the current git commit, captured as the baseline git-sync diffs
    against to detect later code drift; passing it also clears any standing
    code-drift flag (the code has just been re-verified)."""
    component = get_component(graph, component_id)
    component.implemented_version = component.version
    component.implemented_sha = sha
    component.code_drifted = False
    return []


# Edge Functions #
def propose_edge(graph: Graph, edge: Edge) -> list[str]:
    if edge.edge_id in graph.edges:
        return [f"Edge '{edge.edge_id}' already exists"]
    error = validate_edge(graph, edge)
    if error is not None:
        return [error]
    add_edge(graph, edge)
    return []
