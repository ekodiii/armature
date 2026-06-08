import yaml

from models import Component, Edge, EdgeType, FileLocation, Graph, Warning


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
                "edges_in": c.edges_in,
                "edges_out": c.edges_out,
                "children": c.children,
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
        ],
        "warnings": [
            {
                "id": w.id,
                "warning_type": w.warning_type,
                "message": w.message,
                "affected": w.affected,
                "ignored": w.ignored,
                "ignore_reason": w.ignore_reason,
            }
            for w in graph.warnings
        ],
    }
    with open(path, "w") as f:
        yaml.dump(data, f, sort_keys=False)


def load_graph(path: str) -> Graph:
    with open(path) as f:
        data = yaml.safe_load(f)

    graph = Graph.new()

    for cid, c_data in (data.get("components") or {}).items():
        graph.components[cid] = Component(
            component_id=c_data["component_id"],
            description=c_data["description"],
            processing=c_data["processing"],
            input_types=c_data["input_types"],
            output_types=c_data["output_types"],
            z_level=c_data["z_level"],
            external=c_data.get("external", False),
            edges_in=c_data.get("edges_in") or [],
            edges_out=c_data.get("edges_out") or [],
            children=c_data.get("children") or [],
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

    for e_data in data.get("edges") or []:
        edge = Edge(
            edge_type=EdgeType(e_data["edge_type"]),
            from_id=e_data["from_id"],
            to_id=e_data["to_id"],
        )
        graph.edges[edge.edge_id] = edge

    for w_data in data.get("warnings") or []:
        graph.warnings.append(Warning(
            id=w_data["id"],
            warning_type=w_data["warning_type"],
            message=w_data["message"],
            affected=w_data.get("affected") or [],
            ignored=w_data.get("ignored", False),
            ignore_reason=w_data.get("ignore_reason"),
        ))

    return graph
