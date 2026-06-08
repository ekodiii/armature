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


# Component Functions #
def propose_component(graph: Graph, component: Component) -> list[str]:
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


# Edge Functions #
def propose_edge(graph: Graph, edge: Edge) -> list[str]:
    if edge.edge_id in graph.edges:
        return [f"Edge '{edge.edge_id}' already exists"]
    error = validate_edge(graph, edge)
    if error is not None:
        return [error]
    add_edge(graph, edge)
    return []
