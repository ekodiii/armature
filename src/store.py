from models import Component, Edge, EdgeType, Graph


# Component Functions #
def add_component(graph: Graph, component: Component) -> Graph:
    graph.components[component.component_id] = component
    sync_component_fields(graph, component)
    return graph


def remove_component(graph: Graph, component: Component) -> Graph:
    for edge_id in component.edges_in + component.edges_out:
        edge = get_edge(graph, edge_id)
        _detach_edge(graph, edge)
        graph.edges.pop(edge_id)
    _detach_component(graph, component)
    graph.components.pop(component.component_id)
    return graph


def get_component(graph: Graph, component_id: str) -> Component:
    component = graph.components.get(component_id)
    if component is None:
        raise KeyError(f"Component '{component_id}' not found in graph")
    return component


# Edge Functions #
def add_edge(graph: Graph, edge: Edge) -> Graph:
    graph.edges[edge.edge_id] = edge
    sync_edge_fields(graph, edge)
    return graph


def remove_edge(graph: Graph, edge: Edge) -> Graph:
    _detach_edge(graph, edge)
    graph.edges.pop(edge.edge_id)
    return graph


def get_edge(graph: Graph, edge_id: str) -> Edge:
    edge = graph.edges.get(edge_id)
    if edge is None:
        raise KeyError(f"Edge '{edge_id}' not found in graph")
    return edge


# Sync Functions #
# Wires a newly added component to its parent: updates the parent's children
# list and materializes the SCOPE edge so traversal and the edge dict agree.
def sync_component_fields(graph: Graph, component: Component):
    if component.parent_id is not None:
        get_component(graph, component.parent_id).children.append(
            component.component_id
        )
        add_edge(graph, Edge(EdgeType.SCOPE, component.parent_id, component.component_id))


# Updates edges_in and edges_out on the relevant components
def sync_edge_fields(graph: Graph, edge: Edge):
    get_component(graph, edge.from_id).edges_out.append(edge.edge_id)
    get_component(graph, edge.to_id).edges_in.append(edge.edge_id)


def _detach_edge(graph: Graph, edge: Edge):
    get_component(graph, edge.from_id).edges_out.remove(edge.edge_id)
    get_component(graph, edge.to_id).edges_in.remove(edge.edge_id)


def _detach_component(graph: Graph, component: Component):
    if component.parent_id is not None:
        get_component(graph, component.parent_id).children.remove(
            component.component_id
        )
