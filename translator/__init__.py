import sys
from pathlib import Path
from typing import Callable, Optional

_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models import Graph  # noqa: E402
from serializer import load_graph  # noqa: E402

from .graph_commit import commit  # noqa: E402
from .graph_synthesis import synthesize  # noqa: E402
from .llm_annotation import annotate  # noqa: E402
from .models import GraphDraft, MigrationReport  # noqa: E402
from .source_ingestion import ingest  # noqa: E402
from .static_analysis import analyze  # noqa: E402


def run(
    source_root: str,
    graph: Graph,
    graph_path: str,
    annotate_fn: Optional[Callable[[dict], dict]] = None,
) -> MigrationReport:
    """Full pipeline: source → structure → annotate → synthesize → commit.

    annotate_fn receives a context dict per node and returns
    {input_types, output_types, processing}. Pass None to use the heuristic fallback.
    When driven by a Claude agent, the agent supplies annotate_fn by reasoning inline.
    """
    manifest = ingest(source_root)
    structure = analyze(manifest)
    annotated = annotate(structure, manifest, annotate_fn)
    draft = synthesize(annotated, source_root)
    return commit(draft, graph, graph_path)


def run_from_path(
    source_root: str,
    graph_path: str,
    annotate_fn: Optional[Callable[[dict], dict]] = None,
) -> MigrationReport:
    """Convenience: load graph from YAML, run pipeline, save back."""
    try:
        graph = load_graph(graph_path)
    except FileNotFoundError:
        graph = Graph.new()
    return run(source_root, graph, graph_path, annotate_fn)


__all__ = [
    "run",
    "run_from_path",
    "ingest",
    "analyze",
    "annotate",
    "synthesize",
    "commit",
    "GraphDraft",
    "MigrationReport",
]
