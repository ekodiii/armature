import yaml

from models import Component, Edge, EdgeType, FileLocation, Graph, Warning


# Only authoritative state is written. Derived fields are recomputed on load:
#   - edges_in / edges_out / children come from the edge list + parent_id
#   - SCOPE edges are regenerated from parent_id, so they are not serialized
#   - warnings are recomputed; only the ignore decisions need to persist
def save_graph(graph: Graph, path: str):
    data = {
        "components": {
            cid: {
                "component_id": c.component_id,
                "description": c.description,
                "processing": c.processing,
                "input_types": c.input_types,
                "output_types": c.output_types,
                "z_level": c.z_level,
                "external": c.external,
                "parent_id": c.parent_id,
                "version": c.version,
                "locations": [
                    {
                        "path": loc.path,
                        "start_line": loc.start_line,
                        "end_line": loc.end_line,
                    }
                    for loc in c.locations
                ],
            }
            for cid, c in graph.components.items()
        },
        "edges": [
            {
                "edge_type": e.edge_type.value,
                "from_id": e.from_id,
                "to_id": e.to_id,
            }
            for e in graph.edges.values()
            if e.edge_type != EdgeType.SCOPE
        ],
        "warnings": [
            {"id": w.id, "ignore_reason": w.ignore_reason}
            for w in graph.warnings
            if w.ignored
        ],
    }
    with open(path, "w") as f:
        yaml.dump(data, f, sort_keys=False)


def load_graph(path: str) -> Graph:
    with open(path) as f:
        data = yaml.safe_load(f)

    graph = Graph.new()

    # Components carry no derived fields; edges_in/out/children start empty and
    # are rebuilt below. (Older files may include them; they are ignored.)
    for cid, c_data in (data.get("components") or {}).items():
        graph.components[cid] = Component(
            component_id=c_data["component_id"],
            description=c_data["description"],
            processing=c_data["processing"],
            input_types=c_data["input_types"],
            output_types=c_data["output_types"],
            z_level=c_data["z_level"],
            external=c_data.get("external", False),
            parent_id=c_data.get("parent_id"),
            version=c_data.get("version", 0),
            locations=[
                FileLocation(
                    path=loc["path"],
                    start_line=loc.get("start_line"),
                    end_line=loc.get("end_line"),
                )
                for loc in c_data.get("locations") or []
            ],
        )

    # Rebuild the hierarchy (children lists + SCOPE edges) from parent_id.
    for component in graph.components.values():
        if component.parent_id is not None:
            parent = graph.components[component.parent_id]
            parent.children.append(component.component_id)
            scope = Edge(EdgeType.SCOPE, component.parent_id, component.component_id)
            graph.edges[scope.edge_id] = scope
            parent.edges_out.append(scope.edge_id)
            component.edges_in.append(scope.edge_id)

    # Rebuild FLOW / REFERENCE edges. SCOPE edges are derived above, so skip any
    # that an older file may have persisted (avoids duplicates).
    for e_data in data.get("edges") or []:
        if e_data["edge_type"] == EdgeType.SCOPE.value:
            continue
        edge = Edge(
            edge_type=EdgeType(e_data["edge_type"]),
            from_id=e_data["from_id"],
            to_id=e_data["to_id"],
        )
        graph.edges[edge.edge_id] = edge
        graph.components[edge.from_id].edges_out.append(edge.edge_id)
        graph.components[edge.to_id].edges_in.append(edge.edge_id)

    # Only ignored warnings persist; type/message/affected are recomputed by the
    # warning system on the next run. Active warnings are not stored.
    for w_data in data.get("warnings") or []:
        if not w_data.get("ignored", True):
            continue
        graph.warnings.append(Warning(
            id=w_data["id"],
            warning_type=w_data.get("warning_type", ""),
            message=w_data.get("message", ""),
            affected=w_data.get("affected") or [],
            ignored=True,
            ignore_reason=w_data.get("ignore_reason"),
        ))

    return graph
