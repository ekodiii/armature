from models import EdgeType, Graph, Warning
from store import get_component, get_edge
from validator import detect_cycles, validate_input_coverage, validate_output_coverage


# Warning Checks #
def check_hanging_outputs(graph: Graph) -> list[Warning]:
    result = []
    for component in graph.components.values():
        if component.external:
            continue
        consumed = set()
        for edge_id in component.edges_out:
            edge = get_edge(graph, edge_id)
            if edge.edge_type == EdgeType.FLOW:
                for t in get_component(graph, edge.to_id).input_types:
                    consumed.add(t)
        # A child's output that matches a type its parent declares is not
        # hanging — it flows up through the parent's boundary (FLOW is
        # sibling-only, so that consumption is invisible at this level).
        # Without this exemption every exit node warns, which pressures
        # authors to invent fictional sibling edges just to silence it.
        if component.parent_id is not None:
            consumed |= set(get_component(graph, component.parent_id).output_types)
        hanging = [t for t in component.output_types if t not in consumed]
        if hanging:
            result.append(Warning(
                id=f"hanging_output__{component.component_id}",
                warning_type="hanging_output",
                message=(
                    "One or more output types have no downstream consumer. "
                    "This may indicate a missing connection, a required storage component, "
                    "or an intentionally terminal output such as a log or side effect. "
                    "Safe to ignore if intentional."
                ),
                affected=hanging,
            ))
    return result


def check_starved_inputs(graph: Graph) -> list[Warning]:
    result = []
    for component in graph.components.values():
        if component.external:
            continue
        # Only meaningful once a component is wired into sibling flow. A pure
        # entry node with no incoming FLOW is a graph boundary, not a bug.
        flow_in = [
            get_edge(graph, eid)
            for eid in component.edges_in
            if get_edge(graph, eid).edge_type == EdgeType.FLOW
        ]
        if not flow_in:
            continue
        produced = set()
        for edge in flow_in:
            for t in get_component(graph, edge.from_id).output_types:
                produced.add(t)
        starved = [t for t in component.input_types if t not in produced]
        if starved:
            result.append(Warning(
                id=f"starved_input__{component.component_id}",
                warning_type="starved_input",
                message=(
                    "One or more input types are not produced by any upstream component. "
                    "This may indicate a missing connection, or an input supplied at the "
                    "graph boundary. Safe to ignore if this input enters from outside the graph."
                ),
                affected=starved,
            ))
    return result


def check_coverage(graph: Graph) -> list[Warning]:
    result = []
    for component in graph.components.values():
        gaps = (
            validate_input_coverage(graph, component)
            + validate_output_coverage(graph, component)
        )
        if gaps:
            result.append(Warning(
                id=f"coverage_gap__{component.component_id}",
                warning_type="coverage_gap",
                message=(
                    "This component's declared types are not fully covered by its "
                    "entry/exit children. Expected while a level is still being decomposed. "
                    "Safe to ignore mid-authoring; resolve before treating the level as complete."
                ),
                affected=gaps,
            ))
    return result


def check_flow_cycles(graph: Graph) -> list[Warning]:
    cycle = detect_cycles(graph, EdgeType.FLOW)
    if not cycle:
        return []
    # Identify the warning by the cycle's members (order-independent) so ignoring
    # one intentional feedback loop does not also silence a different, accidental
    # cycle elsewhere in the graph.
    cycle_key = "_".join(sorted(set(cycle)))
    return [Warning(
        id=f"flow_cycle__{cycle_key}",
        warning_type="flow_cycle",
        message=(
            "A cycle was detected in the FLOW graph. This may be a legitimate feedback "
            "or retry loop, or an accidental loop. Safe to ignore if intentional."
        ),
        affected=cycle,
    )]


def check_undefined_types(graph: Graph) -> list[Warning]:
    result = []
    for component in graph.components.values():
        if component.external:
            continue
        undefined = []
        if not component.input_types:
            undefined.append("input_types")
        if not component.output_types:
            undefined.append("output_types")
        if undefined:
            result.append(Warning(
                id=f"undefined_types__{component.component_id}",
                warning_type="undefined_types",
                message=(
                    "This component has undefined input or output types. "
                    "Research or planning is required before this component can be decomposed further. "
                    "Safe to ignore if deliberately left abstract at this stage."
                ),
                affected=undefined,
            ))
    return result


def check_orphaned_components(graph: Graph) -> list[Warning]:
    if len(graph.components) <= 1:
        return []
    result = []
    for component in graph.components.values():
        if not component.edges_in and not component.edges_out:
            result.append(Warning(
                id=f"orphaned_component__{component.component_id}",
                warning_type="orphaned_component",
                message=(
                    "This component has no connections to the rest of the graph. "
                    "It may be a new component not yet wired in, or a remnant that should be removed. "
                    "Safe to ignore if still being planned."
                ),
                affected=[component.component_id],
            ))
    return result


# Aggregate #
def run_all_warnings(graph: Graph) -> list[Warning]:
    existing = {w.id: w for w in graph.warnings}

    fresh = (
        check_hanging_outputs(graph)
        + check_starved_inputs(graph)
        + check_undefined_types(graph)
        + check_orphaned_components(graph)
        + check_coverage(graph)
        + check_flow_cycles(graph)
    )

    seen = set()
    result = []
    for w in fresh:
        if w.id in seen:
            continue
        seen.add(w.id)
        prior = existing.get(w.id)
        if prior and prior.ignored:
            w.ignored = prior.ignored
            w.ignore_reason = prior.ignore_reason
        result.append(w)

    return result


# Ignore #
def ignore_warning(warnings: list[Warning], warning_id: str, reason: str) -> list[Warning]:
    for w in warnings:
        if w.id == warning_id:
            w.ignored = True
            w.ignore_reason = reason
            return warnings
    raise KeyError(f"Warning '{warning_id}' not found")


def get_active_warnings(warnings: list[Warning]) -> list[Warning]:
    return [w for w in warnings if not w.ignored]
