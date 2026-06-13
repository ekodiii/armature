"""
translator/lift.py — Trans-Lifter (deterministic preparation + assist layer)

The semantic lift from CodeSkeleton to SemanticModel is performed by an LLM
*operator* driving the MCP authoring tools (propose_component / propose_edge).
Nothing in this module calls an LLM, writes descriptions, or decides
responsibility groupings. Every function here is deterministic: it computes
evidence, slices work into context-sized regions, tracks progress, and checks
the operator's output mechanically. The operator makes the judgments.

Large-codebase model: the skeleton worksheet stays small even when the code is
huge, so translation runs as recursive scope-bounded descent —
  top pass        : author z=0 subsystems from the compact worksheet
  region passes   : one fresh operator session per region against a slice
  coverage loop   : coverage_tracker until empty (progress lives in the graph)
  stitch pass     : reconcile cross-region edges
There is no graph merging; all sessions write into one graph.

One function per trans-lifter child node:
  scope_slicer        (lift-scope-slicer)
  abstraction_planner (lift-abstraction-planner)
  contract_recoverer  (lift-contract-recoverer)
  flow_completer      (lift-flow-completer)
  intent_describer    (lift-intent-describer)
  consolidator        (lift-consolidator)
  coverage_tracker    (lift-coverage-tracker)
  stitch_reconciler   (lift-stitch-reconciler)
Plus the prepare_lift driver that bundles the prep passes for one session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from models import EdgeType, Graph
from translator.skeleton import (
    CallEdge,
    CodeSkeleton,
    DataflowEdge,
    SymbolRecord,
    build_skeleton,
)
from translator.source_ingestion import ingest
from translator.verify import coverage_checker, location_anchor_checker

# A ScopeSpec is one of:
#   None       -> the whole codebase (small-codebase single pass)
#   str        -> a path prefix, e.g. "translator/"
#   set[str]   -> an explicit set of symbol ids
ScopeSpec = Union[None, str, set]

_ATOMIC_KINDS = {"function", "method"}
_LEAF_KINDS = {"function", "method", "class"}

# Return-hint bases that carry no domain meaning on their own — when a symbol's
# I/O is shaped like these, the operator must invent a semantic type name.
_PRIMITIVE_BASES = {
    "dict", "list", "tuple", "set", "frozenset", "str", "int", "float",
    "bool", "bytes", "complex", "none", "nonetype", "any", "object",
}
_CONTAINER_BASES = {"list", "tuple", "set", "frozenset"}

# Method names that mutate a container in place: their effect is invisible to
# the skeleton's def-use dataflow (no value is bound), so the operator must
# wire any resulting FLOW by hand.
_MUTATION_METHODS = {
    "append", "extend", "insert", "add", "update", "setdefault",
    "__setitem__", "appendleft", "extendleft", "put", "push",
}


# ===========================================================================
# Region slicing  (lift-scope-slicer)
# ===========================================================================

@dataclass
class BoundaryStub:
    """Terse summary of an out-of-region symbol referenced by in-region code."""
    id: str
    name: str
    signature: str
    path: str


@dataclass
class RegionSkeleton:
    """One unit of lift work: full detail in-region, terse stubs at the edge."""
    scope: str                              # human label for the slice
    source_root: str                        # so per-symbol source can be read
    in_region_ids: set                      # symbol ids fully in this region
    symbols: list                           # full SymbolRecords, in-region only
    boundary_stubs: list                    # BoundaryStub for referenced externals
    call_edges: list                        # CallEdges with both ends in-region
    boundary_call_edges: list               # CallEdges in-region -> boundary
    dataflow_edges: list                    # DataflowEdges with both ends in-region


def _signature(sym: SymbolRecord) -> str:
    params = ", ".join(sym.params)
    ret = sym.return_hint or "?"
    return f"{sym.name}({params}) -> {ret}"


def _in_region(sym: SymbolRecord, scope: ScopeSpec) -> bool:
    if scope is None:
        return True
    if isinstance(scope, str):
        norm = sym.path.replace("\\", "/")
        return norm.startswith(scope) or norm == scope.rstrip("/")
    # explicit symbol-id set
    return sym.id in scope


def scope_slicer(
    skeleton: CodeSkeleton,
    scope: ScopeSpec,
    source_root: str = ".",
) -> RegionSkeleton:
    """Slice the whole-codebase skeleton to one region.

    In-region symbols keep full detail; out-of-region symbols that in-region
    code references become terse boundary stubs (id, signature, file).
    scope=None degenerates to the whole codebase (single-pass small mode).
    """
    in_region: list[SymbolRecord] = [s for s in skeleton.symbols if _in_region(s, scope)]
    in_ids = {s.id for s in in_region}
    by_id = {s.id: s for s in skeleton.symbols}

    # Intra-region vs boundary call edges
    region_calls: list[CallEdge] = []
    boundary_calls: list[CallEdge] = []
    boundary_ids: set[str] = set()
    for e in skeleton.call_edges:
        if e.caller_id in in_ids:
            if e.callee_id in in_ids:
                region_calls.append(e)
            else:
                boundary_calls.append(e)
                boundary_ids.add(e.callee_id)

    # Dataflow edges with both endpoints in-region
    region_df: list[DataflowEdge] = [
        e for e in skeleton.dataflow_edges
        if e.within in in_ids and e.from_callee in in_ids and e.to_callee in in_ids
    ]

    stubs: list[BoundaryStub] = []
    for bid in sorted(boundary_ids):
        sym = by_id.get(bid)
        if sym is None:
            continue
        stubs.append(BoundaryStub(
            id=sym.id,
            name=sym.name,
            signature=_signature(sym),
            path=sym.path,
        ))

    if scope is None:
        label = "<whole codebase>"
    elif isinstance(scope, str):
        label = scope
    else:
        label = f"<{len(scope)} explicit symbols>"

    return RegionSkeleton(
        scope=label,
        source_root=source_root,
        in_region_ids=in_ids,
        symbols=in_region,
        boundary_stubs=stubs,
        call_edges=region_calls,
        boundary_call_edges=boundary_calls,
        dataflow_edges=region_df,
    )


# ===========================================================================
# Grouping evidence  (lift-abstraction-planner)
# ===========================================================================

@dataclass
class AbstractionContext:
    """A compact grouping worksheet — evidence, not decisions.

    The operator reads this to choose responsibility-driven components; it must
    NOT mirror functions 1:1. Nothing here is a verdict.
    """
    scope: str
    modules: dict                  # module path -> [atomic symbol ids]
    clusters: list                 # [[symbol id, ...]] Leiden communities over call+dataflow
    entrypoints: list              # symbol ids with no in-region caller
    shared_utilities: list         # [(symbol id, fan_in)] fan-in >= threshold
    pure_leaves: list              # symbol ids that call nothing in-region
    fan_in_threshold: int


def _leiden_clusters(atomic_ids: set, pair_weights: dict, resolution: float) -> list:
    """Partition the weighted undirected coupling graph into Leiden communities.

    Connected components are useless as grouping evidence — one bridge call
    fuses two unrelated responsibilities into a single blob (on kova-api the
    whole API collapsed into one 142-symbol cluster). Leiden cuts where
    coupling is sparse, which is exactly the 'minimal nameable interface' a
    component boundary wants."""
    import igraph as ig
    import leidenalg as la

    ids = sorted(atomic_ids)
    if not ids:
        return []
    index = {sid: i for i, sid in enumerate(ids)}
    g = ig.Graph(n=len(ids))
    g.vs["name"] = ids
    if pair_weights:
        g.add_edges([(index[a], index[b]) for a, b in pair_weights])
        g.es["weight"] = list(pair_weights.values())
    part = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights="weight" if pair_weights else None,
        resolution_parameter=resolution,
        n_iterations=2,
        seed=0,  # deterministic worksheets
    )
    clusters = [sorted(g.vs[i]["name"] for i in comm) for comm in part]
    clusters.sort(key=lambda c: (-len(c), c[0] if c else ""))
    return clusters


def abstraction_planner(
    region: RegionSkeleton,
    fan_in_threshold: int = 3,
    resolution: float = 1.0,
) -> AbstractionContext:
    """Compute grouping evidence for a region: per-module symbol lists, Leiden
    communities over the call+dataflow coupling graph, entrypoints, fan-in
    hubs, and pure leaves.

    `resolution` tunes community granularity (higher = smaller, tighter
    clusters); 1.0 is classic modularity."""
    atomic = [s for s in region.symbols if s.kind in _ATOMIC_KINDS]
    atomic_ids = {s.id for s in atomic}

    modules: dict[str, list[str]] = {}
    for s in region.symbols:
        if s.kind in _LEAF_KINDS:
            modules.setdefault(s.path, []).append(s.id)

    # Directed in-region call relationships restricted to atomic symbols
    callers_of: dict[str, set[str]] = {sid: set() for sid in atomic_ids}
    callees_of: dict[str, set[str]] = {sid: set() for sid in atomic_ids}
    for e in region.call_edges:
        if e.caller_id in atomic_ids and e.callee_id in atomic_ids:
            callees_of[e.caller_id].add(e.callee_id)
            callers_of[e.callee_id].add(e.caller_id)

    # Undirected coupling weights: each call edge counts 1; a dataflow edge
    # (one symbol's result consumed by another) counts 2 — passing data is
    # stronger evidence of shared responsibility than a bare call.
    pair_weights: dict[tuple, float] = {}

    def bump(a: str, b: str, w: float) -> None:
        if a == b:
            return
        key = (a, b) if a < b else (b, a)
        pair_weights[key] = pair_weights.get(key, 0.0) + w

    for e in region.call_edges:
        if e.caller_id in atomic_ids and e.callee_id in atomic_ids:
            bump(e.caller_id, e.callee_id, 1.0)
    for e in region.dataflow_edges:
        if e.from_callee in atomic_ids and e.to_callee in atomic_ids:
            bump(e.from_callee, e.to_callee, 2.0)

    clusters = _leiden_clusters(atomic_ids, pair_weights, resolution)

    entrypoints = sorted(sid for sid in atomic_ids if not callers_of[sid])
    pure_leaves = sorted(sid for sid in atomic_ids if not callees_of[sid])
    shared = sorted(
        ((sid, len(callers_of[sid])) for sid in atomic_ids if len(callers_of[sid]) >= fan_in_threshold),
        key=lambda t: (-t[1], t[0]),
    )

    return AbstractionContext(
        scope=region.scope,
        modules=modules,
        clusters=clusters,
        entrypoints=entrypoints,
        shared_utilities=shared,
        pure_leaves=pure_leaves,
        fan_in_threshold=fan_in_threshold,
    )


# ===========================================================================
# Contract recovery  (lift-contract-recoverer)
# ===========================================================================

@dataclass
class ContractHint:
    """Signature-derived I/O hint for one symbol; needs_naming flags erased
    contracts where the operator must invent a semantic type name."""
    symbol_id: str
    name: str
    params: list
    return_hint: Optional[str]
    signature: str
    needs_naming: bool
    reason: str


def _base_of(hint: str) -> str:
    """Outermost type token of an annotation, lowercased (no subscript)."""
    h = hint.strip()
    for sep in ("[", "("):
        if sep in h:
            h = h.split(sep, 1)[0]
    return h.split(".")[-1].strip().lower()


def _names_a_domain_type(hint: str) -> bool:
    """True if the annotation contains a non-primitive, capitalized type name."""
    import re
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", hint):
        if token.lower() in _PRIMITIVE_BASES:
            continue
        if token in ("Optional", "Union", "Sequence", "Iterable", "Mapping", "Iterator"):
            continue
        if token[:1].isupper():
            return True
    return False


def _needs_naming(return_hint: Optional[str]) -> tuple[bool, str]:
    if return_hint is None or not return_hint.strip():
        return True, "no return annotation — output type unknown"
    base = _base_of(return_hint)
    if base == "dict":
        return True, f"dict-shaped output ({return_hint}) — invent a record/type name"
    if base in _CONTAINER_BASES:
        if _names_a_domain_type(return_hint):
            return False, ""
        return True, f"container of primitives ({return_hint}) — name the element type"
    if base in _PRIMITIVE_BASES:
        return True, f"primitive-shaped output ({return_hint}) — invent a semantic type"
    return False, ""


def contract_recoverer(region: RegionSkeleton) -> dict:
    """Surface each in-region symbol's signature types and flag the ones whose
    I/O is dict/primitive-shaped so the operator names the erased contract."""
    hints: dict[str, ContractHint] = {}
    for s in region.symbols:
        if s.kind not in _ATOMIC_KINDS:
            continue
        needs, reason = _needs_naming(s.return_hint)
        hints[s.id] = ContractHint(
            symbol_id=s.id,
            name=s.name,
            params=list(s.params),
            return_hint=s.return_hint,
            signature=_signature(s),
            needs_naming=needs,
            reason=reason,
        )
    return hints


# ===========================================================================
# Flow candidates + blind-spot flags  (lift-flow-completer)
# ===========================================================================

@dataclass
class FlowCandidate:
    """A candidate edge for the operator to confirm, or a blind-spot flag.

    blind_spot=False : producer->consumer edge derived from the skeleton.
    blind_spot=True  : a function where static dataflow is blind (container
                       mutation / accumulation); to_symbol is None and the
                       operator must wire the real FLOW by hand.
    """
    from_symbol: str
    to_symbol: Optional[str]
    kind: str                 # "dataflow" | "call" | "blind_spot"
    blind_spot: bool
    var: Optional[str]
    within: Optional[str]
    note: str


def flow_completer(region: RegionSkeleton) -> list:
    """Emit candidate producer->consumer edges from the skeleton's dataflow and
    call edges, plus blind-spot flags on functions whose container-mutation /
    loop-accumulation call sites defeat static def-use analysis."""
    candidates: list[FlowCandidate] = []

    # Dataflow-derived producer -> consumer (strongest signal)
    seen_df: set[tuple[str, str, str]] = set()
    for e in region.dataflow_edges:
        key = (e.from_callee, e.to_callee, e.var)
        if key in seen_df:
            continue
        seen_df.add(key)
        candidates.append(FlowCandidate(
            from_symbol=e.from_callee,
            to_symbol=e.to_callee,
            kind="dataflow",
            blind_spot=False,
            var=e.var,
            within=e.within,
            note=f"value '{e.var}' produced then consumed inside {e.within}",
        ))

    # Call-derived caller -> callee (weaker; many are control not data flow)
    seen_call: set[tuple[str, str]] = set()
    for e in region.call_edges:
        key = (e.caller_id, e.callee_id)
        if key in seen_call:
            continue
        seen_call.add(key)
        candidates.append(FlowCandidate(
            from_symbol=e.caller_id,
            to_symbol=e.callee_id,
            kind="call",
            blind_spot=False,
            var=None,
            within=None,
            note=f"call site at line {e.lineno}",
        ))

    # Blind spots: in-place container mutation defeats the skeleton's def-use
    # pass (a `.append(x)` binds nothing, so x's flow into the container is
    # invisible). Flagging *every* mutator is noise, so we surface only the
    # genuine hidden flows: a mutation whose argument is a function parameter
    # or a value produced by a prior call in the same function.
    for s in region.symbols:
        if s.kind not in _ATOMIC_KINDS:
            continue
        params = set(s.params)
        produced: set[str] = set()       # vars bound to some call's result
        hidden_args: dict[str, str] = {}  # arg name -> mutation method
        for site in s.call_sites:
            if site.callee_name in _MUTATION_METHODS:
                for arg in site.arg_names:
                    if arg in params or arg in produced:
                        hidden_args.setdefault(arg, site.callee_name)
            for var in site.assigned_to:
                produced.add(var)
        if hidden_args:
            detail = ", ".join(f"{a} via {m}()" for a, m in sorted(hidden_args.items()))
            candidates.append(FlowCandidate(
                from_symbol=s.id,
                to_symbol=None,
                kind="blind_spot",
                blind_spot=True,
                var=None,
                within=s.id,
                note=(
                    f"in-place container mutation hides flow of [{detail}] — "
                    f"static dataflow is blind here; wire downstream FLOW by hand"
                ),
            ))

    return candidates


# ===========================================================================
# Per-symbol describe context  (lift-intent-describer)
# ===========================================================================

@dataclass
class DescribeContext:
    """Thin helper: one symbol's source snippet + call neighbours so the
    operator can write its what/why. No description is generated here."""
    symbol_id: str
    name: str
    kind: str
    path: str
    start_line: int
    end_line: int
    signature: str
    snippet: str
    callers: list           # [(id, name)]
    callees: list           # [(id, name)]


def intent_describer(region: RegionSkeleton, symbol_id: str) -> DescribeContext:
    """Assemble snippet + neighbours for a single symbol from the region."""
    by_id = {s.id: s for s in region.symbols}
    sym = by_id.get(symbol_id)
    if sym is None:
        raise KeyError(f"{symbol_id} not in region '{region.scope}'")

    snippet = ""
    try:
        path = Path(region.source_root) / sym.path
        lines = path.read_text(errors="replace").splitlines()
        snippet = "\n".join(lines[sym.start_line - 1: sym.end_line])
    except OSError:
        snippet = "<source unavailable>"

    name_of = lambda sid: by_id[sid].name if sid in by_id else sid.split("::")[-1]
    callees = sorted(
        {(e.callee_id, name_of(e.callee_id)) for e in region.call_edges if e.caller_id == symbol_id}
    )
    callers = sorted(
        {(e.caller_id, name_of(e.caller_id)) for e in region.call_edges if e.callee_id == symbol_id}
    )

    return DescribeContext(
        symbol_id=sym.id,
        name=sym.name,
        kind=sym.kind,
        path=sym.path,
        start_line=sym.start_line,
        end_line=sym.end_line,
        signature=_signature(sym),
        snippet=snippet,
        callers=callers,
        callees=callees,
    )


# ===========================================================================
# Shared anchoring helpers (used by the post-authoring audits)
# ===========================================================================

def _file_to_symbols(skeleton: CodeSkeleton) -> dict:
    idx: dict[str, list[SymbolRecord]] = {}
    for sym in skeleton.symbols:
        idx.setdefault(sym.path, []).append(sym)
    return idx


def _best_anchor_by_symbol(graph: Graph, skeleton: CodeSkeleton) -> dict:
    """Map each skeleton symbol id -> the most specific (smallest-span)
    non-external component whose location overlaps it."""
    file_idx = _file_to_symbols(skeleton)
    best: dict[str, tuple[int, str]] = {}  # sym_id -> (span, component_id)
    for cid, comp in graph.components.items():
        if comp.external:
            continue
        for loc in comp.locations:
            syms = file_idx.get(loc.path)
            if not syms:
                continue
            start = loc.start_line or 1
            end = loc.end_line or 999999
            span = end - start
            for sym in syms:
                if sym.start_line <= end and sym.end_line >= start:
                    prev = best.get(sym.id)
                    if prev is None or span < prev[0]:
                        best[sym.id] = (span, cid)
    return {sid: cid for sid, (_span, cid) in best.items()}


def _root_of(graph: Graph, component_id: str) -> str:
    """Top-level ancestor (z=0 / no parent) of a component — its subtree root."""
    seen: set[str] = set()
    cur = component_id
    while True:
        comp = graph.components.get(cur)
        if comp is None or comp.parent_id is None or cur in seen:
            return cur
        seen.add(cur)
        cur = comp.parent_id


# ===========================================================================
# Granularity review  (lift-consolidator)
# ===========================================================================

@dataclass
class ResidueFlag:
    """A merge-up candidate: an authored node that merely mirrors one code
    symbol with no contract gain (call-tree residue)."""
    component_id: str
    parent_id: Optional[str]
    detail: str


def consolidator(graph: Graph, skeleton: CodeSkeleton) -> list:
    """Flag call-tree residue: nodes anchored to exactly one symbol with no
    contract gain. Reuses verify.py's degenerate-node detection."""
    state = location_anchor_checker(graph, skeleton)
    state = coverage_checker(state)
    flags: list[ResidueFlag] = []
    for f in state.coverage_findings:
        if f.kind != "degenerate_node":
            continue
        comp = graph.components.get(f.subject)
        flags.append(ResidueFlag(
            component_id=f.subject,
            parent_id=comp.parent_id if comp else None,
            detail=f.detail,
        ))
    return flags


# ===========================================================================
# Coverage / resume mechanism  (lift-coverage-tracker)
# ===========================================================================

@dataclass
class CoverageReport:
    """Uncovered skeleton symbols grouped by module — the resume mechanism.
    Empty (is_complete) means every region is done."""
    by_module: dict                # module path -> [(symbol id, detail)]
    total_uncovered: int
    total_leaf: int
    covered: int
    is_complete: bool


def coverage_tracker(graph: Graph, skeleton: CodeSkeleton) -> CoverageReport:
    """Report skeleton symbols not yet covered by any authored node, grouped by
    module. Reuses verify.py's coverage logic. Drives the next-region loop."""
    state = location_anchor_checker(graph, skeleton)
    state = coverage_checker(state)

    by_module: dict[str, list[tuple[str, str]]] = {}
    uncovered = 0
    for f in state.coverage_findings:
        if f.kind != "uncovered_symbol":
            continue
        uncovered += 1
        path = f.subject.split("::")[0]
        by_module.setdefault(path, []).append((f.subject, f.detail))

    total_leaf = sum(1 for s in skeleton.symbols if s.kind in _LEAF_KINDS)
    return CoverageReport(
        by_module=by_module,
        total_uncovered=uncovered,
        total_leaf=total_leaf,
        covered=total_leaf - uncovered,
        is_complete=(uncovered == 0),
    )


# ===========================================================================
# Cross-region stitching  (lift-stitch-reconciler)
# ===========================================================================

@dataclass
class StitchCandidate:
    """A cross-subtree relationship found in the skeleton but absent from the
    graph — surfaced for operator confirmation as REFERENCE or FLOW. One
    candidate per (from_component, to_component, kind); from_symbol/to_symbol
    hold an exemplar code edge and `occurrences` counts how many back it."""
    from_symbol: str
    to_symbol: str
    from_component: str
    to_component: str
    from_root: str
    to_root: str
    kind: str                 # "call" | "dataflow"
    suggested_edge_type: str  # "REFERENCE" | "FLOW"
    occurrences: int = 1


def stitch_reconciler(graph: Graph, skeleton: CodeSkeleton) -> list:
    """Surface skeleton call/dataflow edges whose endpoint symbols are anchored
    in different subtrees of the authored graph AND whose relationship the
    graph does not already model, for operator confirmation.

    A relationship counts as modeled when a FLOW or REFERENCE edge in the same
    direction exists between the anchored components or any of their ancestors
    — a parent-level FLOW summarizes its children's crossings, so re-listing
    every underlying code edge would be pure noise. Candidates are grouped per
    component pair with an occurrence count instead of one entry per code edge."""
    anchor = _best_anchor_by_symbol(graph, skeleton)
    root_cache: dict[str, str] = {}

    def root(cid: str) -> str:
        if cid not in root_cache:
            root_cache[cid] = _root_of(graph, cid)
        return root_cache[cid]

    def ancestors(cid: str) -> list:
        chain, cur, seen_ids = [], cid, set()
        while cur is not None and cur not in seen_ids:
            seen_ids.add(cur)
            chain.append(cur)
            comp = graph.components.get(cur)
            cur = comp.parent_id if comp else None
        return chain

    existing: set[tuple[str, str]] = {
        (e.from_id, e.to_id)
        for e in graph.edges.values()
        if e.edge_type in (EdgeType.FLOW, EdgeType.REFERENCE)
    }

    modeled_cache: dict[tuple[str, str], bool] = {}

    def already_modeled(ca: str, cb: str) -> bool:
        key = (ca, cb)
        if key not in modeled_cache:
            modeled_cache[key] = any(
                (a, b) in existing for a in ancestors(ca) for b in ancestors(cb)
            )
        return modeled_cache[key]

    grouped: dict[tuple[str, str, str], StitchCandidate] = {}

    def consider(a: str, b: str, kind: str, suggested: str) -> None:
        ca, cb = anchor.get(a), anchor.get(b)
        if ca is None or cb is None or ca == cb:
            return
        ra, rb = root(ca), root(cb)
        if ra == rb:
            return
        if already_modeled(ca, cb):
            return
        key = (ca, cb, kind)
        if key in grouped:
            grouped[key].occurrences += 1
            return
        grouped[key] = StitchCandidate(
            from_symbol=a, to_symbol=b,
            from_component=ca, to_component=cb,
            from_root=ra, to_root=rb,
            kind=kind, suggested_edge_type=suggested,
        )

    for e in skeleton.call_edges:
        consider(e.caller_id, e.callee_id, "call", "REFERENCE")
    for e in skeleton.dataflow_edges:
        consider(e.from_callee, e.to_callee, "dataflow", "FLOW")

    return sorted(
        grouped.values(),
        key=lambda c: (-c.occurrences, c.from_component, c.to_component, c.kind),
    )


# ===========================================================================
# Driver  (trans-lifter)
# ===========================================================================

@dataclass
class LiftContext:
    """Everything one region's operator session needs, bundled by prepare_lift."""
    source_root: str
    skeleton: CodeSkeleton
    region: RegionSkeleton
    abstraction: AbstractionContext
    contracts: dict
    flows: list


def prepare_lift(source_root: str, scope: ScopeSpec = None) -> LiftContext:
    """Ingest + skeleton + the four region prep passes, bundled for one operator
    session. scope=None is the whole-codebase single pass."""
    skeleton = build_skeleton(ingest(source_root), source_root)
    region = scope_slicer(skeleton, scope, source_root=source_root)
    return LiftContext(
        source_root=source_root,
        skeleton=skeleton,
        region=region,
        abstraction=abstraction_planner(region),
        contracts=contract_recoverer(region),
        flows=flow_completer(region),
    )
