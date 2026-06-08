from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RawNode:
    node_id: str          # "relative/path.py::ClassName::method_name"
    kind: str             # "module" | "class" | "function"
    params: list[str]
    return_hint: Optional[str]
    start_line: int
    end_line: int
    relative_path: str
    raw_imports: list[str]
    raw_calls: list[str]
    external: bool = False
    # populated by sa_dependency_resolver
    imports: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    # populated by la_batch_annotator
    input_types: list[str] = field(default_factory=list)
    output_types: list[str] = field(default_factory=list)
    processing: str = ""
    depth: int = 0


@dataclass
class FileRecord:
    relative_path: str
    language: str
    text: str


@dataclass
class DispatchedFiles:
    python: list[FileRecord]
    generic: list[FileRecord]


@dataclass
class StructureMap:
    nodes: list[RawNode]
    call_edges: list[tuple[str, str]]
    import_edges: list[tuple[str, str]]


@dataclass
class WriteLog:
    written: list[str] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)
    edge_written: list[str] = field(default_factory=list)
    edge_failed: list[dict] = field(default_factory=list)
    edge_skipped: list[dict] = field(default_factory=list)
    active_warnings_snapshot: list = field(default_factory=list)


@dataclass
class MigrationReport:
    summary: str
    failures: list[dict]
    skipped_edges: list[dict]
    validation_issues: list[str]
    active_warnings: list


@dataclass
class GraphDraft:
    components: list
    edges: list
    validation_issues: list[str]
    source_root: str
    stats: dict
    issues: list[str] = field(default_factory=list)
