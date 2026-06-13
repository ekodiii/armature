from collections import deque
from typing import Optional

from models import Component, EdgeType, Graph
from store import get_component, get_edge


def rank(graph: Graph, query: str, top_k: int = 10) -> list[tuple[str, float]]:
    """Score each component by relevance to query and return top-k (component_id, score) pairs.

    Scoring weights (case-insensitive substring match):
      3.0  — id contains query
      2.0  — description contains query
      1.0  — processing contains query
    A small structural boost (+0.5) is added for each already-matched neighbor
    (FLOW or SCOPE) so closely wired clusters surface together.
    Returns a list sorted descending by score, capped at top_k."""
    q = query.lower()
    if not q:
        return []

    base: dict[str, float] = {}
    for cid, c in graph.components.items():
        score = 0.0
        if q in cid.lower():
            score += 3.0
        if q in c.description.lower():
            score += 2.0
        if q in c.processing.lower():
            score += 1.0
        if score > 0:
            base[cid] = score

    # Structural proximity boost: components adjacent (any direction, FLOW/SCOPE)
    # to a matched node get a small lift so related clusters surface together.
    boosted: dict[str, float] = dict(base)
    for cid, score in base.items():
        c = graph.components[cid]
        neighbors_ids = set()
        for edge_id in c.edges_in + c.edges_out:
            edge = graph.edges.get(edge_id)
            if edge and edge.edge_type in (EdgeType.FLOW, EdgeType.SCOPE):
                other = edge.from_id if edge.to_id == cid else edge.to_id
                neighbors_ids.add(other)
        for nid in neighbors_ids:
            boosted[nid] = boosted.get(nid, 0.0) + 0.5

    ranked = sorted(boosted.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_k]


def get_neighbors(graph: Graph, component_id: str, edge_type: EdgeType, upstream: bool = False) -> list[Component]:
    component = get_component(graph, component_id)
    edge_ids = component.edges_in if upstream else component.edges_out
    neighbors = []
    for edge_id in edge_ids:
        edge = get_edge(graph, edge_id)
        if edge.edge_type == edge_type:
            neighbor_id = edge.from_id if upstream else edge.to_id
            neighbors.append(get_component(graph, neighbor_id))
    return neighbors


def get_references(graph: Graph, component_id: str, incoming: bool = False) -> list[Component]:
    """Components linked by weak REFERENCE edges. Kept separate from the strict
    FLOW/SCOPE traversals so references never leak into contract logic — they are
    navigation hints (cross-boundary data flow), not part of the graph's structure."""
    component = get_component(graph, component_id)
    edge_ids = component.edges_in if incoming else component.edges_out
    result = []
    for edge_id in edge_ids:
        edge = get_edge(graph, edge_id)
        if edge.edge_type == EdgeType.REFERENCE:
            other_id = edge.from_id if incoming else edge.to_id
            result.append(get_component(graph, other_id))
    return result


def get_subgraph(graph: Graph, component_id: str, depth: int = -1) -> list[Component]:
    result = []
    visited = set()

    def walk(cid: str, remaining: int):
        if cid in visited:
            return
        visited.add(cid)
        component = get_component(graph, cid)
        result.append(component)
        if remaining == 0:
            return
        for edge_id in component.edges_out:
            edge = get_edge(graph, edge_id)
            if edge.edge_type == EdgeType.SCOPE:
                walk(edge.to_id, remaining - 1 if remaining > 0 else remaining)

    walk(component_id, depth)
    return result


def get_path(graph: Graph, from_id: str, to_id: str) -> Optional[list[str]]:
    if from_id == to_id:
        return [from_id]

    queue = deque([[from_id]])
    visited = {from_id}

    while queue:
        path = queue.popleft()
        current_id = path[-1]
        for edge_id in get_component(graph, current_id).edges_out:
            edge = get_edge(graph, edge_id)
            if edge.edge_type != EdgeType.FLOW:
                continue
            next_id = edge.to_id
            if next_id == to_id:
                return path + [next_id]
            if next_id not in visited:
                visited.add(next_id)
                queue.append(path + [next_id])

    return None


def get_impact(graph: Graph, component_id: str) -> dict[str, list[Component]]:
    return {
        "upstream": _walk_flow(graph, component_id, upstream=True),
        "downstream": _walk_flow(graph, component_id, upstream=False),
    }


def _walk_flow(graph: Graph, component_id: str, upstream: bool) -> list[Component]:
    result = []
    visited = {component_id}
    queue = deque([component_id])
    while queue:
        cid = queue.popleft()
        component = get_component(graph, cid)
        edge_ids = component.edges_in if upstream else component.edges_out
        for edge_id in edge_ids:
            edge = get_edge(graph, edge_id)
            if edge.edge_type != EdgeType.FLOW:
                continue
            next_id = edge.from_id if upstream else edge.to_id
            if next_id not in visited:
                visited.add(next_id)
                result.append(get_component(graph, next_id))
                queue.append(next_id)
    return result
