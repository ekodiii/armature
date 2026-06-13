from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# Edges
class EdgeType(Enum):
    FLOW = "FLOW"            # data movement between siblings, same z-level
    SCOPE = "SCOPE"          # parent to direct child, one z-level down
    REFERENCE = "REFERENCE"  # weak annotation link, may cross parents/z-levels


@dataclass
class Edge:
    edge_type: EdgeType
    from_id: str
    to_id: str

    @property
    def edge_id(self) -> str:
        return f"{self.from_id}__{self.to_id}__{self.edge_type.value}"


# Components
@dataclass
class Component:
    # metadata
    component_id: str
    description: str
    processing: str
    input_types: list[str]
    output_types: list[str]
    z_level: int

    # boundary node — atomic, never decomposed
    external: bool = False

    # connectors
    edges_in: list[str] = field(default_factory=list)
    edges_out: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    locations: list["FileLocation"] = field(default_factory=list)

    # bumped on every validated update — optimistic concurrency primitive
    version: int = 0
    # the `version` whose spec the code was last implemented against.
    # None = never implemented. < version = spec changed since, code is stale.
    implemented_version: Optional[int] = None
    # the git commit HEAD was at when mark_implemented last ran — the baseline
    # git-sync diffs against to detect that the code moved under this node.
    implemented_sha: Optional[str] = None
    # set by a reconcile pass when the anchored code changed since implemented_sha,
    # cleared on the next mark_implemented. This is the one staleness signal that
    # cannot be derived from in-graph state alone — git is the external oracle, so
    # reconcile materializes it. Drives the 'drifted' status.
    code_drifted: bool = False
    # None = part of the as-built base (the running system). A non-None value names
    # the planned FEATURE this node belongs to: a proposal layered beside the base
    # that does not yet exist in code. Base views (warnings, orient, stats, the
    # impl queue) exclude feature nodes, so planned work never pollutes the live
    # graph. A node sheds its feature tag — joining the base — when its code lands.
    feature: Optional[str] = None


# Warnings
@dataclass
class Warning:
    id: str
    warning_type: str
    message: str
    affected: list[str]
    ignored: bool = False
    ignore_reason: Optional[str] = None


# DA GRAPHHHH
@dataclass
class Graph:
    components: dict[str, Component] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)
    warnings: list[Warning] = field(default_factory=list)
    # commit a `reconcile` last synced this graph to; the default baseline for
    # drift detection on components that have no implemented_sha of their own.
    last_synced_sha: Optional[str] = None

    @staticmethod
    def new() -> "Graph":
        return Graph()


# File Locations
@dataclass
class FileLocation:
    path: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
