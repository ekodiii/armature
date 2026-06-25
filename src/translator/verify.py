"""
translator/verify.py — Anchor Verifier

Verifies every semantic claim in an Armature Graph against a CodeSkeleton so
the model adds meaning without drifting into fiction.

Five pipeline stages, each threading a VerificationState:
  1. location_anchor_checker       — anchors → real symbols / line ranges
  2. coverage_checker              — symbols covered; nodes grounded; degenerate nodes
  3. contract_consistency_checker  — type contracts chain along FLOW edges
  4. flow_grounding_checker        — FLOW/REFERENCE edges backed by skeleton relationships
  5. discrepancy_reconciler        — collects findings into VerifiedModel + trust score

Top-level entry point: verify(graph, skeleton) -> VerifiedModel
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from models import EdgeType, Graph
from translator.skeleton import CodeSkeleton, SymbolRecord


# ---------------------------------------------------------------------------
# Shared data types
# ---------------------------------------------------------------------------

@dataclass
class AnchorFinding:
    """A single hallucinated or misplaced location anchor."""
    component_id: str
    location_path: str
    start_line: Optional[int]
    end_line: Optional[int]
    reason: str   # "file_not_in_skeleton" | "no_symbol_overlap" | "range_outside_file"


@dataclass
class CoverageFinding:
    """A symbol with no covering node, or a node with no real symbol anchor."""
    kind: str              # "uncovered_symbol" | "anchorless_node" | "degenerate_node"
    subject: str           # symbol id or component_id
    detail: str


@dataclass
class ContractFinding:
    """A type mismatch on a FLOW edge."""
    edge_id: str
    from_id: str
    to_id: str
    producer_output: list[str]
    consumer_input: list[str]
    detail: str


@dataclass
class GroundingFinding:
    """A FLOW or REFERENCE edge with no backing in the skeleton.

    kind distinguishes the two cases so scoring never has to parse `detail`:
      "unverifiable" — neither endpoint is anchored, so the edge cannot be
                       checked (excluded from the grounding denominator).
      "ungrounded"   — both endpoints anchored but no skeleton link connects
                       them (a real finding against the score).
    """
    edge_id: str
    from_id: str
    to_id: str
    edge_type: str
    detail: str
    kind: str = "ungrounded"


@dataclass
class PrecisionFinding:
    """A leaf node that 'covers' many files/symbols with no decomposition.

    This is the inclusion-only blob pattern: anchoring one node to whole files
    makes every symbol inside count as 'covered' without being modeled, which
    both fakes coverage and defeats edge grounding (any-symbol-to-any-symbol
    matching over a huge set almost always finds an incidental link).
    """
    component_id: str
    files: int
    covered_symbols: int
    detail: str


@dataclass
class ExternalFinding:
    """A real external boundary (third-party dependency) with no external node.

    A non-trivial codebase that imports third-party modules but models ZERO
    external boundaries has mislabeled every I/O edge as internal.
    """
    module: str
    importer_components: list[str]
    detail: str


@dataclass
class VerificationState:
    """Accumulated verification results threaded through all stages."""
    graph: Graph
    skeleton: CodeSkeleton
    anchor_findings: list[AnchorFinding] = field(default_factory=list)
    coverage_findings: list[CoverageFinding] = field(default_factory=list)
    contract_findings: list[ContractFinding] = field(default_factory=list)
    grounding_findings: list[GroundingFinding] = field(default_factory=list)
    precision_findings: list[PrecisionFinding] = field(default_factory=list)
    external_findings: list[ExternalFinding] = field(default_factory=list)


@dataclass
class VerifiedModel:
    """Final output: the graph, the skeleton, and a structured trust report."""
    graph: Graph
    skeleton: CodeSkeleton
    anchor_findings: list[AnchorFinding]
    coverage_findings: list[CoverageFinding]
    contract_findings: list[ContractFinding]
    grounding_findings: list[GroundingFinding]
    precision_findings: list[PrecisionFinding]
    external_findings: list[ExternalFinding]
    trust_score: float           # 0.0–1.0
    trust_breakdown: dict        # sub-scores for each dimension

    def __repr__(self) -> str:
        lines = [
            f"VerifiedModel(trust_score={self.trust_score:.2%})",
            f"  anchor_findings     : {len(self.anchor_findings)}",
            f"  coverage_findings   : {len(self.coverage_findings)}",
            f"  contract_findings   : {len(self.contract_findings)}",
            f"  grounding_findings  : {len(self.grounding_findings)}",
            f"  precision_findings  : {len(self.precision_findings)}",
            f"  external_findings   : {len(self.external_findings)}",
            f"  trust_breakdown     : {self.trust_breakdown}",
        ]
        if self.anchor_findings:
            lines.append("  -- sample anchor findings --")
            for f_ in self.anchor_findings[:5]:
                lines.append(f"    [{f_.component_id}] {f_.location_path}:{f_.start_line}-{f_.end_line} — {f_.reason}")
        if self.coverage_findings:
            lines.append("  -- sample coverage findings --")
            for f_ in self.coverage_findings[:5]:
                lines.append(f"    [{f_.kind}] {f_.subject}: {f_.detail}")
        if self.contract_findings:
            lines.append("  -- sample contract findings --")
            for f_ in self.contract_findings[:5]:
                lines.append(f"    [{f_.edge_id}] {f_.from_id}→{f_.to_id}: {f_.detail}")
        if self.grounding_findings:
            lines.append("  -- sample grounding findings --")
            for f_ in self.grounding_findings[:5]:
                lines.append(f"    [{f_.edge_type}] {f_.from_id}→{f_.to_id}: {f_.detail}")
        if self.precision_findings:
            lines.append("  -- sample precision findings (coarse blobs) --")
            for f_ in self.precision_findings[:5]:
                lines.append(f"    [{f_.component_id}] {f_.files} files / {f_.covered_symbols} symbols")
        if self.external_findings:
            lines.append("  -- sample external findings (unmodeled boundaries) --")
            for f_ in self.external_findings[:5]:
                lines.append(f"    [{f_.module}] importers={f_.importer_components}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 1 — location_anchor_checker
# ---------------------------------------------------------------------------

def location_anchor_checker(graph: Graph, skeleton: CodeSkeleton) -> VerificationState:
    """Check every non-external component's location anchors against the skeleton.

    For each declared location:
      - The file (path) must appear in at least one skeleton symbol.
      - If start_line/end_line are given, at least one symbol in that file must
        overlap the declared range (symbol.start_line <= end_line and
        symbol.end_line >= start_line).

    Initialises and returns a fresh VerificationState.
    """
    state = VerificationState(graph=graph, skeleton=skeleton)

    # Build per-file symbol index: path -> [SymbolRecord]
    file_to_symbols: dict[str, list[SymbolRecord]] = {}
    for sym in skeleton.symbols:
        file_to_symbols.setdefault(sym.path, []).append(sym)

    known_paths = set(file_to_symbols.keys())

    for cid, comp in graph.components.items():
        if comp.external:
            continue
        for loc in comp.locations:
            path = loc.path
            if path not in known_paths:
                state.anchor_findings.append(AnchorFinding(
                    component_id=cid,
                    location_path=path,
                    start_line=loc.start_line,
                    end_line=loc.end_line,
                    reason="file_not_in_skeleton",
                ))
                continue

            # If no line range given, existence of the file is sufficient
            if loc.start_line is None and loc.end_line is None:
                continue

            start = loc.start_line or 1
            end = loc.end_line or start

            if start > end:
                state.anchor_findings.append(AnchorFinding(
                    component_id=cid,
                    location_path=path,
                    start_line=loc.start_line,
                    end_line=loc.end_line,
                    reason="range_outside_file",
                ))
                continue

            # Check that at least one symbol in the file overlaps the range
            symbols_in_file = file_to_symbols[path]
            overlaps = any(
                sym.start_line <= end and sym.end_line >= start
                for sym in symbols_in_file
            )
            if not overlaps:
                state.anchor_findings.append(AnchorFinding(
                    component_id=cid,
                    location_path=path,
                    start_line=loc.start_line,
                    end_line=loc.end_line,
                    reason="no_symbol_overlap",
                ))

    return state


# ---------------------------------------------------------------------------
# Stage 2 — coverage_checker
# ---------------------------------------------------------------------------

def coverage_checker(state: VerificationState) -> VerificationState:
    """Check symbol coverage and node grounding.

    Three sub-checks:
      (a) Symbols not claimed by any component location → missed code.
      (b) Non-external nodes with no location anchor → fiction / anchorless.
      (c) Degenerate nodes: anchored to exactly one atomic symbol (function/method)
          with no contract gain vs. the skeleton (same types, no grouping benefit).
    """
    graph = state.graph
    skeleton = state.skeleton

    # Build per-file symbol index
    file_to_symbols: dict[str, list[SymbolRecord]] = {}
    for sym in skeleton.symbols:
        file_to_symbols.setdefault(sym.path, []).append(sym)

    # --- Collect claimed symbols (symbol ids covered by at least one component) ---
    # A component claims a symbol if its location overlaps that symbol's range.
    claimed_symbol_ids: set[str] = set()
    # Map component_id -> list of claimed symbol ids
    comp_claimed: dict[str, list[str]] = {}

    for cid, comp in graph.components.items():
        if comp.external:
            continue
        claimed_for_comp: list[str] = []
        for loc in comp.locations:
            path = loc.path
            if path not in file_to_symbols:
                continue
            start = loc.start_line or 1
            end = loc.end_line or 999999
            for sym in file_to_symbols[path]:
                if sym.start_line <= end and sym.end_line >= start:
                    claimed_symbol_ids.add(sym.id)
                    claimed_for_comp.append(sym.id)
        comp_claimed[cid] = claimed_for_comp

    # (a) Uncovered symbols — only flag leaf symbols (functions/methods/classes),
    # not module-level records (those are structural, not semantic entities).
    leaf_kinds = {"function", "method", "class"}
    for sym in skeleton.symbols:
        if sym.kind not in leaf_kinds:
            continue
        if sym.id not in claimed_symbol_ids:
            state.coverage_findings.append(CoverageFinding(
                kind="uncovered_symbol",
                subject=sym.id,
                detail=f"{sym.kind} {sym.name} in {sym.path}:{sym.start_line}-{sym.end_line} has no covering node",
            ))

    # (b) Anchorless non-external nodes (no locations at all)
    for cid, comp in graph.components.items():
        if comp.external:
            continue
        if not comp.locations:
            state.coverage_findings.append(CoverageFinding(
                kind="anchorless_node",
                subject=cid,
                detail=f"component '{cid}' has no location anchors — potential fiction",
            ))

    # (c) Degenerate nodes: anchored to exactly one atomic symbol, and that symbol
    # is claimed by no other component either (i.e. pure pass-through, no grouping).
    # We only flag functions/methods with no children and no meaningful contract.
    by_id = {s.id: s for s in skeleton.symbols}
    for cid, comp in graph.components.items():
        if comp.external:
            continue
        claimed = comp_claimed.get(cid, [])
        # Filter to atomic (non-module) symbols
        atomic = [
            s for s in claimed
            if (sr := by_id.get(s)) is not None and sr.kind in ("function", "method")
        ]
        if len(atomic) == 1 and not comp.children:
            sym_id = atomic[0]
            sym = by_id.get(sym_id)
            if sym is not None:
                # Degenerate if the component's type contract adds no information
                # beyond the raw symbol (both input and output are generic or empty)
                types_trivial = (
                    len(comp.input_types) <= 1 and len(comp.output_types) <= 1
                )
                if types_trivial:
                    state.coverage_findings.append(CoverageFinding(
                        kind="degenerate_node",
                        subject=cid,
                        detail=(
                            f"component '{cid}' wraps single symbol {sym_id} "
                            f"with trivial contract — potential call-tree residue"
                        ),
                    ))

    return state


# ---------------------------------------------------------------------------
# Stage 3 — contract_consistency_checker
# ---------------------------------------------------------------------------

def contract_consistency_checker(state: VerificationState) -> VerificationState:
    """Check that type contracts chain along FLOW edges.

    For each FLOW edge (producer → consumer): producer.output_types and
    consumer.input_types must share at least one type string. Records breaks.
    """
    graph = state.graph

    for edge_id, edge in graph.edges.items():
        if edge.edge_type != EdgeType.FLOW:
            continue

        producer = graph.components.get(edge.from_id)
        consumer = graph.components.get(edge.to_id)
        if producer is None or consumer is None:
            continue

        producer_out = set(producer.output_types)
        consumer_in = set(consumer.input_types)

        if not producer_out or not consumer_in:
            # One side is untyped — flag as a gap but not a hard break
            if not producer_out and not consumer_in:
                continue  # both untyped, no info to check
            state.contract_findings.append(ContractFinding(
                edge_id=edge_id,
                from_id=edge.from_id,
                to_id=edge.to_id,
                producer_output=list(producer_out),
                consumer_input=list(consumer_in),
                detail=(
                    f"one side has no declared types: "
                    f"producer_out={list(producer_out)} consumer_in={list(consumer_in)}"
                ),
            ))
            continue

        # Check for shared type (case-insensitive for robustness)
        prod_lower = {t.lower() for t in producer_out}
        cons_lower = {t.lower() for t in consumer_in}
        if not prod_lower & cons_lower:
            state.contract_findings.append(ContractFinding(
                edge_id=edge_id,
                from_id=edge.from_id,
                to_id=edge.to_id,
                producer_output=list(producer_out),
                consumer_input=list(consumer_in),
                detail=(
                    f"type mismatch: producer outputs {sorted(producer_out)} "
                    f"but consumer expects {sorted(consumer_in)}"
                ),
            ))

    return state


# ---------------------------------------------------------------------------
# Stage 4 — flow_grounding_checker
# ---------------------------------------------------------------------------

def flow_grounding_checker(state: VerificationState) -> VerificationState:
    """Check that each FLOW/REFERENCE edge has a real relationship in the skeleton.

    Strategy (loose — semantic edges aggregate many code edges):
      1. Collect the skeleton symbol ids anchored by each endpoint component.
      2. For a pair (from_syms, to_syms), check if any call edge or dataflow edge
         exists between any symbol in from_syms and any in to_syms (or vice versa
         for REFERENCE which is undirected).
      3. If neither endpoint has any skeleton anchor, we cannot verify → flag.
      4. If one endpoint is external, skip (cross-boundary by design).
    """
    graph = state.graph
    skeleton = state.skeleton

    # Build per-file symbol index and per-component claimed-symbol set
    file_to_symbols: dict[str, list[SymbolRecord]] = {}
    for sym in skeleton.symbols:
        file_to_symbols.setdefault(sym.path, []).append(sym)

    def claimed_ids(comp_id: str) -> set[str]:
        comp = graph.components.get(comp_id)
        if comp is None or comp.external:
            return set()
        ids: set[str] = set()
        for loc in comp.locations:
            path = loc.path
            if path not in file_to_symbols:
                continue
            start = loc.start_line or 1
            end = loc.end_line or 999999
            for sym in file_to_symbols[path]:
                if sym.start_line <= end and sym.end_line >= start:
                    ids.add(sym.id)
        return ids

    # Build skeleton relationship sets for fast lookup
    # call edges: (caller_id, callee_id)
    call_pairs: set[tuple[str, str]] = set()
    for ce in skeleton.call_edges:
        call_pairs.add((ce.caller_id, ce.callee_id))

    # dataflow edges: (from_callee, to_callee)
    df_pairs: set[tuple[str, str]] = set()
    for de in skeleton.dataflow_edges:
        df_pairs.add((de.from_callee, de.to_callee))

    def has_backing(from_syms: set[str], to_syms: set[str], bidirectional: bool = False) -> bool:
        """True if any skeleton relationship connects any from_sym to any to_sym."""
        for f in from_syms:
            for t in to_syms:
                if (f, t) in call_pairs or (f, t) in df_pairs:
                    return True
                # Also check: to_sym imports / includes from_sym (module-level)
                # A module-level SymbolRecord covering the callee counts as a link.
                # (Catches cases where a component aggregates an entire module.)
                if bidirectional:
                    if (t, f) in call_pairs or (t, f) in df_pairs:
                        return True
        return False

    def has_import_link(from_syms: set[str], to_syms: set[str]) -> bool:
        """True if any file in from_syms imports any file in to_syms or vice versa."""
        from_paths = {s.split("::")[0] for s in from_syms}
        to_paths = {s.split("::")[0] for s in to_syms}
        for ri in skeleton.resolved:
            if ri.resolved_path:
                if ri.importer_file in from_paths and ri.resolved_path in to_paths:
                    return True
                if ri.importer_file in to_paths and ri.resolved_path in from_paths:
                    return True
        return False

    for edge_id, edge in graph.edges.items():
        if edge.edge_type not in (EdgeType.FLOW, EdgeType.REFERENCE):
            continue

        from_comp = graph.components.get(edge.from_id)
        to_comp = graph.components.get(edge.to_id)
        if from_comp is None or to_comp is None:
            continue

        # Skip cross-boundary edges (at least one external endpoint)
        if from_comp.external or to_comp.external:
            continue

        from_syms = claimed_ids(edge.from_id)
        to_syms = claimed_ids(edge.to_id)

        # If neither side is anchored we can't verify → flag as unverifiable
        if not from_syms and not to_syms:
            state.grounding_findings.append(GroundingFinding(
                edge_id=edge_id,
                from_id=edge.from_id,
                to_id=edge.to_id,
                edge_type=edge.edge_type.value,
                detail="neither endpoint has skeleton anchors — edge unverifiable",
                kind="unverifiable",
            ))
            continue

        # If one side is unanchored, we can only do a partial check
        if not from_syms or not to_syms:
            # Partial — don't flag, but note it could be improved
            continue

        bidirectional = (edge.edge_type == EdgeType.REFERENCE)
        grounded = (
            has_backing(from_syms, to_syms, bidirectional=bidirectional)
            or has_import_link(from_syms, to_syms)
        )

        if not grounded:
            state.grounding_findings.append(GroundingFinding(
                edge_id=edge_id,
                from_id=edge.from_id,
                to_id=edge.to_id,
                edge_type=edge.edge_type.value,
                detail=(
                    f"no call, dataflow, or import link found between "
                    f"{len(from_syms)} from-symbols and {len(to_syms)} to-symbols"
                ),
                kind="ungrounded",
            ))

    return state


# ---------------------------------------------------------------------------
# Stage — precision_checker  (coarse inclusion-only leaf nodes)
# ---------------------------------------------------------------------------

# A leaf node anchored to more than this many files, or covering more than this
# many leaf symbols, is an under-decomposed blob: it fakes coverage (every
# symbol in the whole file counts as 'covered') and makes edge grounding
# meaningless. Calibrated so hand-authored graphs with focused leaves pass while
# whole-subpackage blobs (e.g. one node over 20-30 files) are flagged.
COARSE_FILE_THRESHOLD = 4
COARSE_SYMBOL_THRESHOLD = 40


def precision_checker(state: VerificationState) -> VerificationState:
    """Flag leaf components that 'cover' many files/symbols with no decomposition.

    Only LEAF nodes (no children) are judged: a parent is expected to span its
    whole subtree, but a leaf with no children that still rakes in dozens of
    files is modeling by inclusion, not by responsibility.
    """
    graph = state.graph
    skeleton = state.skeleton
    leaf_kinds = {"function", "method", "class"}

    file_to_symbols: dict[str, list[SymbolRecord]] = {}
    for sym in skeleton.symbols:
        file_to_symbols.setdefault(sym.path, []).append(sym)

    for cid, comp in graph.components.items():
        if comp.external or comp.children or not comp.locations:
            continue
        files = {loc.path for loc in comp.locations}
        covered: set[str] = set()
        for loc in comp.locations:
            start = loc.start_line or 1
            end = loc.end_line or 999999
            for s in file_to_symbols.get(loc.path, []):
                if s.kind in leaf_kinds and s.start_line <= end and s.end_line >= start:
                    covered.add(s.id)
        if len(files) > COARSE_FILE_THRESHOLD or len(covered) > COARSE_SYMBOL_THRESHOLD:
            state.precision_findings.append(PrecisionFinding(
                component_id=cid,
                files=len(files),
                covered_symbols=len(covered),
                detail=(
                    f"leaf component '{cid}' anchors {len(files)} files / "
                    f"{len(covered)} symbols with no decomposition — inclusion-only "
                    f"blob, not a modeled responsibility (coverage and grounding "
                    f"over this node are not trustworthy)"
                ),
            ))

    return state


# ---------------------------------------------------------------------------
# Stage — externals_checker  (unmodeled external boundaries)
# ---------------------------------------------------------------------------

def externals_checker(state: VerificationState) -> VerificationState:
    """Flag real external boundaries that are not modeled as external nodes.

    Conservative by design: it fires only when the codebase imports third-party
    modules (genuine external dependencies, not stdlib) yet the graph contains
    ZERO external components — the gross failure mode. Per-boundary coverage
    (matching each third-party module to a specific external node) is left to a
    future refinement; module names rarely match node ids reliably.
    """
    graph = state.graph
    skeleton = state.skeleton

    thirdparty: dict[str, set[str]] = {}
    for sc in skeleton.scope_classes:
        if sc.scope == "third-party" and sc.external_module:
            thirdparty.setdefault(sc.external_module, set()).add(sc.importer_file)

    if not thirdparty:
        return state
    if any(c.external for c in graph.components.values()):
        return state  # graph models boundaries; can't assess partial coverage by name

    file_to_comps: dict[str, list[str]] = {}
    for cid, comp in graph.components.items():
        if comp.external:
            continue
        for loc in (comp.locations or []):
            file_to_comps.setdefault(loc.path, []).append(cid)

    for mod, importers in sorted(thirdparty.items(), key=lambda kv: -len(kv[1]))[:15]:
        comps = sorted({cid for f in importers for cid in file_to_comps.get(f, [])})
        state.external_findings.append(ExternalFinding(
            module=mod,
            importer_components=comps[:5],
            detail=(
                f"third-party dependency '{mod}' is a real external boundary "
                f"(imported by {len(importers)} file(s)) but the graph models NO "
                f"external node for it"
            ),
        ))

    return state


# ---------------------------------------------------------------------------
# Stage — discrepancy_reconciler
# ---------------------------------------------------------------------------

def discrepancy_reconciler(state: VerificationState) -> VerifiedModel:
    """Collect all findings and compute a structured trust report.

    Trust score = weighted average of four sub-scores:
      - anchor_score   : fraction of non-external components with no anchor finding
      - coverage_score : fraction of leaf symbols that are covered
      - contract_score : fraction of FLOW edges with no contract break
      - grounding_score: fraction of FLOW/REF edges (both anchored) that are grounded

    LLM-assisted reconciliation is operator-driven; this function only produces
    the report. A clean VerificationState passes through unchanged.
    """
    graph = state.graph
    skeleton = state.skeleton

    # --- Anchor score ---
    non_ext = [c for c in graph.components.values() if not c.external]
    comps_with_anchor_findings = {f.component_id for f in state.anchor_findings}
    if non_ext:
        anchor_score = 1.0 - len(comps_with_anchor_findings) / len(non_ext)
    else:
        anchor_score = 1.0

    # --- Coverage score (leaf symbol coverage) ---
    leaf_kinds = {"function", "method", "class"}
    leaf_symbols = [s for s in skeleton.symbols if s.kind in leaf_kinds]
    uncovered = sum(1 for f in state.coverage_findings if f.kind == "uncovered_symbol")
    if leaf_symbols:
        coverage_score = 1.0 - uncovered / len(leaf_symbols)
    else:
        coverage_score = 1.0

    # --- Contract score ---
    flow_edges = [e for e in graph.edges.values() if e.edge_type == EdgeType.FLOW]
    if flow_edges:
        contract_score = 1.0 - len(state.contract_findings) / len(flow_edges)
    else:
        contract_score = 1.0

    # --- Grounding score ---
    ref_flow_edges = [
        e for e in graph.edges.values()
        if e.edge_type in (EdgeType.FLOW, EdgeType.REFERENCE)
    ]
    # Only count verifiable edges (both endpoints anchored and non-external) in
    # the denominator, and only count ungrounded findings (not the unverifiable
    # ones we already excluded) in the numerator — otherwise the two disagree and
    # the score is understated, even negative.
    unverifiable = sum(1 for f in state.grounding_findings if f.kind == "unverifiable")
    ungrounded = sum(1 for f in state.grounding_findings if f.kind == "ungrounded")
    verifiable = len(ref_flow_edges) - unverifiable
    if verifiable > 0:
        grounding_score = max(0.0, 1.0 - ungrounded / verifiable)
    else:
        grounding_score = 1.0

    # --- Precision score (1 - prevalence of coarse inclusion-only leaves) ---
    non_ext_leaves = [
        c for c in graph.components.values()
        if not c.external and not c.children and c.locations
    ]
    if non_ext_leaves:
        precision_score = 1.0 - len(state.precision_findings) / len(non_ext_leaves)
    else:
        precision_score = 1.0
    precision_score = max(0.0, min(1.0, precision_score))

    # --- Externals score (are real boundaries represented at all?) ---
    externals_score = 0.0 if state.external_findings else 1.0

    # Weighted composite. precision + externals are the hardening dimensions:
    # they catch coarse inclusion-only modeling and mislabeled boundaries that
    # the anchor / coverage / grounding checks structurally miss.
    trust_score = (
        0.20 * anchor_score
        + 0.20 * coverage_score
        + 0.15 * precision_score
        + 0.15 * contract_score
        + 0.20 * grounding_score
        + 0.10 * externals_score
    )
    trust_score = max(0.0, min(1.0, trust_score))

    trust_breakdown = {
        "anchor_score": round(anchor_score, 3),
        "coverage_score": round(coverage_score, 3),
        "precision_score": round(precision_score, 3),
        "contract_score": round(contract_score, 3),
        "grounding_score": round(grounding_score, 3),
        "externals_score": round(externals_score, 3),
        "non_external_components": len(non_ext),
        "non_external_leaves": len(non_ext_leaves),
        "leaf_symbols_in_skeleton": len(leaf_symbols),
        "flow_edges": len(flow_edges),
        "flow_ref_edges": len(ref_flow_edges),
        "anchor_findings": len(state.anchor_findings),
        "coverage_findings": len(state.coverage_findings),
        "contract_findings": len(state.contract_findings),
        "grounding_findings": len(state.grounding_findings),
        "precision_findings": len(state.precision_findings),
        "external_findings": len(state.external_findings),
    }

    return VerifiedModel(
        graph=state.graph,
        skeleton=state.skeleton,
        anchor_findings=state.anchor_findings,
        coverage_findings=state.coverage_findings,
        contract_findings=state.contract_findings,
        grounding_findings=state.grounding_findings,
        precision_findings=state.precision_findings,
        external_findings=state.external_findings,
        trust_score=round(trust_score, 3),
        trust_breakdown=trust_breakdown,
    )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def verify(graph: Graph, skeleton: CodeSkeleton) -> VerifiedModel:
    """Run all five verification stages in order and return a VerifiedModel.

    Args:
        graph:    An Armature Graph (semantic model), loaded via load_graph().
        skeleton: A CodeSkeleton from build_skeleton(), representing real code.

    Returns:
        VerifiedModel with trust_score (0–1) and structured findings per stage.
    """
    state = location_anchor_checker(graph, skeleton)
    state = coverage_checker(state)
    state = precision_checker(state)
    state = contract_consistency_checker(state)
    state = flow_grounding_checker(state)
    state = externals_checker(state)
    return discrepancy_reconciler(state)
