from typing import Optional

from models import Component, Edge, EdgeType, Graph
from store import get_component, get_edge


def validate_edge(graph: Graph, edge: Edge) -> Optional[str]:
    if edge.from_id not in graph.components:
        return f"Component '{edge.from_id}' does not exist"
    if edge.to_id not in graph.components:
        return f"Component '{edge.to_id}' does not exist"
    if edge.from_id == edge.to_id:
        return f"Edge connects a component to itself: {edge.from_id}"

    from_c = get_component(graph, edge.from_id)
    to_c = get_component(graph, edge.to_id)

    if edge.edge_type == EdgeType.FLOW:
        # FLOW moves data between siblings at the same z-level. No diagonals.
        if from_c.z_level != to_c.z_level:
            return (
                f"FLOW edge connects different z_levels: "
                f"{edge.from_id}({from_c.z_level}) -> {edge.to_id}({to_c.z_level})"
            )
        if from_c.parent_id != to_c.parent_id:
            return (
                f"FLOW edge connects non-siblings: {edge.from_id} and {edge.to_id} "
                f"have different parents ({from_c.parent_id} vs {to_c.parent_id})"
            )
        return None

    if edge.edge_type == EdgeType.REFERENCE:
        # Weak annotation link: the only edge allowed to cross parents and
        # z-levels. Preserves data-flow traceability across a decomposition
        # boundary (e.g. A2 -> B1 when A->B exists one level up). Not a
        # contract — ignored by coverage, cycle, and FLOW-walk logic.
        return None

    # SCOPE descends exactly one level, from a parent to its direct child.
    if to_c.z_level != from_c.z_level + 1:
        return (
            f"SCOPE edge must descend exactly one z_level: "
            f"{edge.from_id}({from_c.z_level}) -> {edge.to_id}({to_c.z_level})"
        )
    if to_c.parent_id != from_c.component_id:
        return (
            f"SCOPE edge must connect a parent to its direct child: "
            f"{edge.to_id}'s parent is {to_c.parent_id}, not {edge.from_id}"
        )
    return None


def validate_component(graph: Graph, component: Component) -> list[str]:
    errors = []
    if component.parent_id is None:
        if component.z_level != 0:
            errors.append(
                f"Root component '{component.component_id}' must be z_level 0, "
                f"got {component.z_level}"
            )
        return errors

    if component.parent_id not in graph.components:
        errors.append(f"Parent '{component.parent_id}' does not exist")
        return errors

    parent = get_component(graph, component.parent_id)
    if parent.external:
        errors.append(
            f"Cannot add child to external component '{parent.component_id}'"
        )
    if component.z_level != parent.z_level + 1:
        errors.append(
            f"Child z_level must be parent z_level + 1: "
            f"'{component.component_id}'({component.z_level}) under "
            f"'{parent.component_id}'({parent.z_level})"
        )
    return errors


# Cross-checks that the denormalized fields (parent_id, children, edges_in,
# edges_out) agree with the authoritative edge dict. Used in tests/debugging.
def validate_consistency(graph: Graph) -> list[str]:
    errors = []
    for edge in graph.edges.values():
        if edge.edge_id not in get_component(graph, edge.from_id).edges_out:
            errors.append(f"Edge '{edge.edge_id}' missing from {edge.from_id}.edges_out")
        if edge.edge_id not in get_component(graph, edge.to_id).edges_in:
            errors.append(f"Edge '{edge.edge_id}' missing from {edge.to_id}.edges_in")
        if edge.edge_type == EdgeType.SCOPE:
            child = get_component(graph, edge.to_id)
            if child.parent_id != edge.from_id:
                errors.append(
                    f"SCOPE edge '{edge.edge_id}' disagrees with "
                    f"{edge.to_id}.parent_id ({child.parent_id})"
                )
            if edge.to_id not in get_component(graph, edge.from_id).children:
                errors.append(
                    f"SCOPE edge '{edge.edge_id}' has no matching entry in "
                    f"{edge.from_id}.children"
                )
    for component in graph.components.values():
        if component.parent_id is not None:
            scope_id = f"{component.parent_id}__{component.component_id}__SCOPE"
            if scope_id not in graph.edges:
                errors.append(
                    f"'{component.component_id}'.parent_id set but SCOPE edge "
                    f"'{scope_id}' is missing"
                )
    return errors


# Children with no incoming FLOW from siblings are entry nodes;
# children with no outgoing FLOW to siblings are exit nodes.
def _entry_exit_nodes(graph: Graph, component: Component) -> tuple[set[str], set[str]]:
    sibling_ids = set(component.children)
    entry_nodes = set()
    exit_nodes = set()
    for child_id in component.children:
        child = get_component(graph, child_id)
        has_sibling_in = False
        for eid in child.edges_in:
            edge = get_edge(graph, eid)
            if edge.edge_type == EdgeType.FLOW and edge.from_id in sibling_ids:
                has_sibling_in = True
                break
        has_sibling_out = False
        for eid in child.edges_out:
            edge = get_edge(graph, eid)
            if edge.edge_type == EdgeType.FLOW and edge.to_id in sibling_ids:
                has_sibling_out = True
                break
        if not has_sibling_in:
            entry_nodes.add(child_id)
        if not has_sibling_out:
            exit_nodes.add(child_id)
    return entry_nodes, exit_nodes


def validate_input_coverage(graph: Graph, component: Component) -> list[str]:
    if component.external:
        if component.children:
            return [f"External component '{component.component_id}' cannot have children"]
        return []

    if not component.children:
        return []

    entry_nodes, _ = _entry_exit_nodes(graph, component)
    unmatched = set(component.input_types)
    for child_id in entry_nodes:
        for t in get_component(graph, child_id).input_types:
            unmatched.discard(t)
    return [f"Input type '{t}' not covered by entry nodes" for t in unmatched]


def validate_output_coverage(graph: Graph, component: Component) -> list[str]:
    if component.external:
        if component.children:
            return [f"External component '{component.component_id}' cannot have children"]
        return []

    if not component.children:
        return []

    _, exit_nodes = _entry_exit_nodes(graph, component)
    unmatched = set(component.output_types)
    for child_id in exit_nodes:
        for t in get_component(graph, child_id).output_types:
            unmatched.discard(t)
    return [f"Output type '{t}' not covered by exit nodes" for t in unmatched]


def detect_cycles(graph: Graph, edge_type: EdgeType) -> list[str]:
    """Return the node ids forming the first cycle of `edge_type`, in order, or
    [] if the graph is acyclic. Iterative (no recursion-depth limit on deep
    chains) three-colour DFS; reconstructs the actual cycle path rather than
    just the re-entry node, and keeps no state across roots."""
    WHITE, GREY, BLACK = 0, 1, 2
    color: dict[str, int] = {cid: WHITE for cid in graph.components}
    parent: dict[str, str] = {}

    def successors(cid: str) -> list[str]:
        return [
            get_edge(graph, eid).to_id
            for eid in get_component(graph, cid).edges_out
            if get_edge(graph, eid).edge_type == edge_type
        ]

    for root in graph.components:
        if color[root] != WHITE:
            continue
        # Each stack frame is [node, its successors, next-index].
        color[root] = GREY
        stack: list[list] = [[root, successors(root), 0]]
        while stack:
            node, succs, i = stack[-1]
            if i == len(succs):
                color[node] = BLACK
                stack.pop()
                continue
            stack[-1][2] = i + 1
            nxt = succs[i]
            if color[nxt] == WHITE:
                color[nxt] = GREY
                parent[nxt] = node
                stack.append([nxt, successors(nxt), 0])
            elif color[nxt] == GREY:
                # Back edge node -> nxt closes a cycle; walk parents up to nxt.
                cycle = [node]
                cur = node
                while cur != nxt and cur in parent:
                    cur = parent[cur]
                    cycle.append(cur)
                cycle.reverse()
                return cycle

    return []
