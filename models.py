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

    @staticmethod
    def new() -> "Graph":
        return Graph()


# File Locations
@dataclass
class FileLocation:
    path: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
