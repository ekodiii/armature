"""reader.py — Load the armature registry and parse YAML graphs into GraphSnapshot dicts.

Reuses serializer.load_graph and graph_warnings.run_all_warnings from the sibling
src/ modules; no YAML parsing is duplicated here.
"""

import json
import os
from typing import Optional

# These imports work when the server is run from src/ (pythonpath = ["."]).
from serializer import load_graph
from graph_warnings import run_all_warnings

ARMATURE_HOME = os.environ.get("ARMATURE_HOME", os.path.expanduser("~/.armature"))
REGISTRY_PATH = os.path.join(ARMATURE_HOME, "registry.json")


def load_registry() -> dict:
    """Return {"active": name | None, "graphs": {name: path}}."""
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {"active": None, "graphs": {}}


def _status(c) -> str:
    if c.implemented_version is None:
        return "planned"
    if c.code_drifted:
        return "drifted"
    if c.implemented_version == c.version:
        return "implemented"
    return "stale"


def read_graph(name: str, path: str) -> dict:
    """Load the YAML at *path*, run warnings, and return a GraphSnapshot dict."""
    graph = load_graph(path)
    graph.warnings = run_all_warnings(graph)

    # Build component map
    components: dict = {}
    for cid, c in graph.components.items():
        components[cid] = {
            "component_id": c.component_id,
            "description": c.description,
            "processing": c.processing,
            "input_types": c.input_types,
            "output_types": c.output_types,
            "z_level": c.z_level,
            "external": c.external,
            "status": _status(c),
            "parent_id": c.parent_id,
            "children": list(c.children),
            "edges_in": list(c.edges_in),
            "edges_out": list(c.edges_out),
        }

    # Edge list
    edges = [
        {
            "id": edge.edge_id,
            "from_id": edge.from_id,
            "to_id": edge.to_id,
            "edge_type": edge.edge_type.value,
        }
        for edge in graph.edges.values()
    ]

    # Active (non-ignored) warnings only
    warnings = [
        {
            "id": w.id,
            "warning_type": w.warning_type,
            "affected": w.affected,
        }
        for w in graph.warnings
        if not w.ignored
    ]

    # Stats
    status_counts: dict[str, int] = {"planned": 0, "implemented": 0, "stale": 0, "drifted": 0}
    max_z = 0
    for c in graph.components.values():
        st = _status(c)
        if st in status_counts:
            status_counts[st] += 1
        if c.z_level > max_z:
            max_z = c.z_level

    stats = {
        "total": len(graph.components),
        **status_counts,
        "max_z_level": max_z,
    }

    return {
        "graph_name": name,
        "graph_path": path,
        "components": components,
        "edges": edges,
        "warnings": warnings,
        "stats": stats,
    }
