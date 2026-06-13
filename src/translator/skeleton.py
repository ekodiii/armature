"""
translator/skeleton.py — Mechanical Skeleton

Faithful, location-pinned, semantics-free anchor for the semantic-translator
graph.  Six pipeline stages, one per graph node, assembled by build_skeleton().
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .models import FileRecord


# ---------------------------------------------------------------------------
# Data types (one per stage; each embeds the prior so the assembler has all)
# ---------------------------------------------------------------------------

@dataclass
class CallSite:
    """One call expression inside a function body."""
    callee_name: str          # raw textual name (e.g. "ingest", "Path", "os.path.join")
    assigned_to: list[str]    # vars the result is bound to (may be empty)
    arg_names: list[str]      # vars/literals passed as args (best-effort names)
    lineno: int


@dataclass
class ImportRecord:
    """One import statement in a source file."""
    importer_file: str
    module: str               # the module being imported, e.g. "os" or ".models"
    names: list[str]          # names imported (["*"] for star, [] for bare import)
    level: int                # relative import depth (0 = absolute)
    lineno: int


@dataclass
class SymbolRecord:
    """One symbol (module/class/function/method) extracted from source."""
    id: str                   # "path/to/file.py::ClassName::method_name"
    kind: str                 # "module" | "class" | "function" | "method"
    name: str
    path: str                 # relative file path
    start_line: int
    end_line: int
    params: list[str]
    return_hint: Optional[str]
    parent_id: Optional[str]  # enclosing symbol id, or None for module
    raw_call_names: list[str] # raw callee names from AST walk
    call_sites: list[CallSite]  # ordered call sites (only populated for functions)


@dataclass
class SymbolTable:
    """Stage 1 output: all symbols + per-file imports."""
    symbols: list[SymbolRecord]
    imports: list[ImportRecord]


# ---------------------------------------------------------------------------

@dataclass
class ResolvedImport:
    """One import with its resolution result."""
    importer_file: str
    module: str
    names: list[str]
    level: int
    lineno: int
    resolved_path: Optional[str]    # relative file path if resolved to a source file
    external_module: Optional[str]  # top-level module name if not a source file


@dataclass
class ResolvedImports:
    """Stage 2 output: SymbolTable + resolution info for every import."""
    symbol_table: SymbolTable
    resolved: list[ResolvedImport]


# ---------------------------------------------------------------------------

@dataclass
class ScopeClass:
    """Scope classification for one resolved import."""
    importer_file: str
    module: str
    resolved_path: Optional[str]
    external_module: Optional[str]
    scope: str  # "in-scope" | "first-party-out-of-scope" | "stdlib" | "third-party"


@dataclass
class ScopedSymbols:
    """Stage 3 output: ResolvedImports + per-import scope classifications."""
    resolved_imports: ResolvedImports
    scope_classes: list[ScopeClass]
    # external boundary: modules that are not in-scope (stdlib/third-party/fp-oos)
    boundary: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------

@dataclass
class CallEdge:
    """A directed caller → callee edge in the call graph."""
    caller_id: str   # symbol id of the calling function
    callee_id: str   # symbol id of the callee (in-scope)
    lineno: int


@dataclass
class CallGraph:
    """Stage 4 output: ScopedSymbols + resolved call edges."""
    scoped: ScopedSymbols
    edges: list[CallEdge]


# ---------------------------------------------------------------------------

@dataclass
class DataflowEdge:
    """Intra-function def-use edge: result of one call feeds arg of another."""
    within: str       # function symbol id where this happens
    from_callee: str  # callee id whose return value is consumed
    to_callee: str    # callee id that receives it as an argument
    var: str          # variable name that carries the value


@dataclass
class DataflowGraph:
    """Stage 5 output: CallGraph + forward def-use dataflow edges."""
    call_graph: CallGraph
    dataflow_edges: list[DataflowEdge]


# ---------------------------------------------------------------------------

@dataclass
class CodeSkeleton:
    """Stage 6 output: fully assembled skeleton artifact."""
    dataflow_graph: DataflowGraph
    # flattened convenience views
    symbols: list[SymbolRecord]
    imports: list[ImportRecord]
    resolved: list[ResolvedImport]
    scope_classes: list[ScopeClass]
    boundary: set[str]
    call_edges: list[CallEdge]
    dataflow_edges: list[DataflowEdge]


# ===========================================================================
# Stage 1 — symbol_extractor
# ===========================================================================

def _params_from_args(args: ast.arguments) -> list[str]:
    return [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]


def _return_hint(node: ast.FunctionDef | ast.AsyncFunctionDef) -> Optional[str]:
    if node.returns is None:
        return None
    try:
        return ast.unparse(node.returns)
    except Exception:
        return None


def _raw_call_names(body_node: ast.AST) -> list[str]:
    names = []
    for child in ast.walk(body_node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                names.append(func.id)
            elif isinstance(func, ast.Attribute):
                names.append(func.attr)
    return names


def _arg_names_of_call(call_node: ast.Call) -> list[str]:
    """Best-effort: collect Name ids used as positional/keyword args."""
    names = []
    for arg in call_node.args:
        if isinstance(arg, ast.Name):
            names.append(arg.id)
        elif isinstance(arg, ast.Starred) and isinstance(arg.value, ast.Name):
            names.append(arg.value.id)
    for kw in call_node.keywords:
        if isinstance(kw.value, ast.Name):
            names.append(kw.value.id)
    return names


def _build_assign_map(func_node: ast.AST) -> dict[int, list[str]]:
    """Map id(rhs_expression) → [variable names it is assigned to], for every
    assignment in the function body. Built once per function so call-site
    resolution is a dict lookup instead of an O(calls × nodes) re-walk."""
    assign_map: dict[int, list[str]] = {}
    for node in ast.walk(func_node):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    assign_map.setdefault(id(node.value), []).append(tgt.id)
                elif isinstance(tgt, ast.Tuple):
                    for elt in tgt.elts:
                        if isinstance(elt, ast.Name):
                            assign_map.setdefault(id(node.value), []).append(elt.id)
        elif isinstance(node, ast.AnnAssign):
            if node.value is not None and isinstance(node.target, ast.Name):
                assign_map.setdefault(id(node.value), []).append(node.target.id)
        elif isinstance(node, ast.NamedExpr):
            assign_map.setdefault(id(node.value), []).append(node.target.id)
    return assign_map


def _call_sites_from_func(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[CallSite]:
    """Extract ordered CallSite records from a function's AST node."""
    sites: list[CallSite] = []
    assign_map = _build_assign_map(func_node)
    # Walk in source order: ast.walk is not ordered, so we need a visitor
    class _Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if isinstance(func, ast.Name):
                callee = func.id
            elif isinstance(func, ast.Attribute):
                callee = func.attr
            else:
                self.generic_visit(node)
                return
            sites.append(CallSite(
                callee_name=callee,
                assigned_to=assign_map.get(id(node), []),
                arg_names=_arg_names_of_call(node),
                lineno=node.lineno,
            ))
            self.generic_visit(node)

    _Visitor().visit(func_node)
    return sites


def _extract_imports_from_tree(tree: ast.AST, rel_path: str) -> list[ImportRecord]:
    records = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                records.append(ImportRecord(
                    importer_file=rel_path,
                    module=alias.name,
                    names=[],
                    level=0,
                    lineno=node.lineno,
                ))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            records.append(ImportRecord(
                importer_file=rel_path,
                module=module,
                names=[alias.name for alias in node.names],
                level=node.level,
                lineno=node.lineno,
            ))
    return records


def _extract_symbols_from_file(file: FileRecord) -> tuple[list[SymbolRecord], list[ImportRecord]]:
    try:
        tree = ast.parse(file.text, filename=file.relative_path)
    except SyntaxError:
        return [], []

    imports = _extract_imports_from_tree(tree, file.relative_path)
    lines = file.text.splitlines()
    symbols: list[SymbolRecord] = []

    module_id = file.relative_path
    symbols.append(SymbolRecord(
        id=module_id,
        kind="module",
        name=file.relative_path,
        path=file.relative_path,
        start_line=1,
        end_line=len(lines),
        params=[],
        return_hint=None,
        parent_id=None,
        raw_call_names=[],
        call_sites=[],
    ))

    for top in ast.iter_child_nodes(tree):
        if isinstance(top, ast.ClassDef):
            class_id = f"{file.relative_path}::{top.name}"
            symbols.append(SymbolRecord(
                id=class_id,
                kind="class",
                name=top.name,
                path=file.relative_path,
                start_line=top.lineno,
                end_line=getattr(top, "end_lineno", top.lineno),
                params=[],
                return_hint=None,
                parent_id=module_id,
                raw_call_names=_raw_call_names(top),
                call_sites=[],
            ))
            for member in ast.iter_child_nodes(top):
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_id = f"{file.relative_path}::{top.name}::{member.name}"
                    symbols.append(SymbolRecord(
                        id=method_id,
                        kind="method",
                        name=member.name,
                        path=file.relative_path,
                        start_line=member.lineno,
                        end_line=getattr(member, "end_lineno", member.lineno),
                        params=_params_from_args(member.args),
                        return_hint=_return_hint(member),
                        parent_id=class_id,
                        raw_call_names=_raw_call_names(member),
                        call_sites=_call_sites_from_func(member),
                    ))

        elif isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_id = f"{file.relative_path}::{top.name}"
            symbols.append(SymbolRecord(
                id=func_id,
                kind="function",
                name=top.name,
                path=file.relative_path,
                start_line=top.lineno,
                end_line=getattr(top, "end_lineno", top.lineno),
                params=_params_from_args(top.args),
                return_hint=_return_hint(top),
                parent_id=module_id,
                raw_call_names=_raw_call_names(top),
                call_sites=_call_sites_from_func(top),
            ))

    return symbols, imports


def symbol_extractor(manifest: list[FileRecord]) -> SymbolTable:
    """Parse each Python file via AST and extract all symbols with locations,
    signatures, call sites, and import statements into a flat SymbolTable."""
    all_symbols: list[SymbolRecord] = []
    all_imports: list[ImportRecord] = []

    for file in manifest:
        if file.language == "python":
            syms, imps = _extract_symbols_from_file(file)
            all_symbols.extend(syms)
            all_imports.extend(imps)
        else:
            # Non-Python: emit a minimal module record so the file appears in scope
            all_symbols.append(SymbolRecord(
                id=file.relative_path,
                kind="module",
                name=file.relative_path,
                path=file.relative_path,
                start_line=1,
                end_line=len(file.text.splitlines()),
                params=[],
                return_hint=None,
                parent_id=None,
                raw_call_names=[],
                call_sites=[],
            ))

    return SymbolTable(symbols=all_symbols, imports=all_imports)


# ===========================================================================
# Stage 2 — import_resolver
# ===========================================================================

def _resolve_module_to_file(
    module: str,
    level: int,
    importer_file: str,
    source_root: str,
) -> Optional[str]:
    """
    Resolve a Python import to a file path relative to source_root, matching the
    paths on SymbolRecords (which ingest() produces source_root-relative). This
    alignment is what lets call/dataflow resolution and scope classification key
    off the resolved path; returning a CWD-relative path here (the old behaviour)
    silently broke every cross-file link whenever source_root was not ".".
    Falls back to a CWD-relative path for files resolved outside source_root.

    For `from X import Y` the *module* is X, not "X.Y": we resolve only the
    module; the imported names are irrelevant to locating the file.

    Relative imports (level > 0): level=1 means current package, level=2 parent, etc.
    The importer_file is source_root-relative, so we reconstruct the on-disk path
    as source_root/importer_file before walking package levels.
    """
    root = Path(source_root)
    # Full path of the importer relative to CWD
    full_importer = root / importer_file
    pkg_dir = full_importer.parent  # package dir of the importing file (CWD-relative)

    candidates: list[Path] = []

    if level > 0:
        # Relative import: walk up (level - 1) package levels from pkg_dir
        anchor = pkg_dir
        for _ in range(level - 1):
            anchor = anchor.parent
        if module:
            base = anchor / module.replace(".", "/")
        else:
            base = anchor
        candidates.append(base / "__init__.py")
        candidates.append(Path(str(base) + ".py"))
    else:
        # Absolute import. Try, in order: under source_root (imports relative to
        # the scoped dir), under its parent (a package imported by its own name
        # when source_root IS that package), then the CWD repo root. For a
        # repo-root source_root, root.parent collapses to ".", so this is a
        # superset of the previous behaviour.
        if not module:
            return None
        mod_path = module.replace(".", "/")
        for search_root in (root, root.parent, Path(".")):
            base = search_root / mod_path
            candidates.append(base / "__init__.py")
            candidates.append(Path(str(base) + ".py"))

    for path in candidates:
        if path.exists():
            # Prefer source_root-relative (matches SymbolRecord.path); fall back
            # to CWD-relative for first-party files resolved outside source_root.
            for base in (root, Path(".")):
                try:
                    return str(path.relative_to(base))
                except ValueError:
                    continue
            return str(path)

    return None


def import_resolver(symbol_table: SymbolTable, source_root: str) -> ResolvedImports:
    """Resolve each import statement to a source file or external module name.

    Critical bug fix: `from X import Y` resolves module X to its file,
    NOT X/Y.py — the importer file's package is used for relative imports."""
    resolved: list[ResolvedImport] = []

    for imp in symbol_table.imports:
        resolved_path = _resolve_module_to_file(
            imp.module, imp.level, imp.importer_file, source_root
        )
        if resolved_path is not None:
            resolved.append(ResolvedImport(
                importer_file=imp.importer_file,
                module=imp.module,
                names=imp.names,
                level=imp.level,
                lineno=imp.lineno,
                resolved_path=resolved_path,
                external_module=None,
            ))
            # `from pkg import name` where name is itself a submodule: the
            # import binds pkg/name.py, not a symbol in pkg/__init__.py. Emit
            # an extra resolution per such name so call/grounding analysis
            # sees the real file dependency (e.g. `from api.routes import
            # activities` links the importer to api/routes/activities.py).
            if resolved_path.replace("\\", "/").endswith("__init__.py"):
                for name in imp.names:
                    if name == "*":
                        continue
                    sub = _resolve_module_to_file(
                        f"{imp.module}.{name}" if imp.module else name,
                        imp.level, imp.importer_file, source_root,
                    )
                    if sub is not None:
                        resolved.append(ResolvedImport(
                            importer_file=imp.importer_file,
                            module=f"{imp.module}.{name}" if imp.module else name,
                            names=[],
                            level=imp.level,
                            lineno=imp.lineno,
                            resolved_path=sub,
                            external_module=None,
                        ))
        else:
            # External: the top-level module name
            top_level = imp.module.split(".")[0] if imp.module else ""
            resolved.append(ResolvedImport(
                importer_file=imp.importer_file,
                module=imp.module,
                names=imp.names,
                level=imp.level,
                lineno=imp.lineno,
                resolved_path=None,
                external_module=top_level or None,
            ))

    return ResolvedImports(symbol_table=symbol_table, resolved=resolved)


# ===========================================================================
# Stage 3 — scope_classifier
# ===========================================================================

def scope_classifier(resolved: ResolvedImports, source_root: str) -> ScopedSymbols:
    """Classify each resolved import as in-scope, first-party-out-of-scope,
    stdlib (via sys.stdlib_module_names — no hardcoded whitelist), or third-party.

    Anything resolved to a source file is in-scope (or first-party-out-of-scope
    if outside source_root). Unresolved imports are stdlib if their top-level
    name is in sys.stdlib_module_names, else third-party. We deliberately do NOT
    probe the running server's site-packages — that describes Armature's own
    environment, not the analyzed project's — so unknown externals are simply
    treated as third-party (they land on the boundary either way)."""
    stdlib_names: frozenset[str] = getattr(sys, "stdlib_module_names", frozenset())
    # A file is in-scope iff it was actually ingested (i.e. it backs a symbol).
    # This is exact, where a path-prefix test is not, now that resolved paths are
    # source_root-relative like symbol paths.
    in_scope_files = {s.path.replace("\\", "/") for s in resolved.symbol_table.symbols}
    scope_classes: list[ScopeClass] = []
    boundary: set[str] = set()

    for r in resolved.resolved:
        if r.resolved_path is not None:
            # Resolved to a file — in-scope or first-party-out-of-scope
            norm = r.resolved_path.replace("\\", "/")
            if norm in in_scope_files:
                scope = "in-scope"
            else:
                scope = "first-party-out-of-scope"
                boundary.add(r.resolved_path)
        else:
            top = r.external_module or ""
            scope = "stdlib" if top in stdlib_names else "third-party"
            if top:
                boundary.add(top)

        scope_classes.append(ScopeClass(
            importer_file=r.importer_file,
            module=r.module,
            resolved_path=r.resolved_path,
            external_module=r.external_module,
            scope=scope,
        ))

    return ScopedSymbols(
        resolved_imports=resolved,
        scope_classes=scope_classes,
        boundary=boundary,
    )


# ===========================================================================
# Stage 4 — call_graph_builder
# ===========================================================================

def _build_name_resolver(scoped: ScopedSymbols):
    """Return resolve(callee_name, caller_symbol) -> Optional[callee_id].

    Resolution honours real Python name binding instead of matching a bare name
    against every symbol in the codebase (the old first-match-anywhere bug, which
    fabricated edges whenever two symbols shared a name like `run`/`get`):

      1. a top-level function/class defined in the caller's own file
      2. a name imported into the caller's file (from the resolved import target)
      3. a method, only when the name is unambiguous — preferring a method on the
         caller's own class (handles `self.helper()`), else a globally unique
         method name. Ambiguous method names resolve to nothing (we would rather
         miss an edge than invent one).

    Aliased imports (`import x as y`) bind under the original name only, since the
    skeleton's ImportRecord does not capture asname — a known minor gap.
    """
    symbols = scoped.resolved_imports.symbol_table.symbols
    by_id = {s.id: s for s in symbols}
    module_ids = {s.id for s in symbols if s.kind == "module"}

    # file path -> {top-level name: symbol id}
    toplevel_by_file: dict[str, dict[str, str]] = {}
    for s in symbols:
        if s.kind in ("function", "class") and s.parent_id in module_ids:
            toplevel_by_file.setdefault(s.path, {})[s.name] = s.id

    # method name -> [symbol ids], for the unambiguous-fallback path
    methods_by_name: dict[str, list[str]] = {}
    for s in symbols:
        if s.kind == "method":
            methods_by_name.setdefault(s.name, []).append(s.id)

    # importer file -> {imported name: symbol id in the resolved target file}
    imported_names: dict[str, dict[str, str]] = {}
    for ri in scoped.resolved_imports.resolved:
        if ri.resolved_path is None:
            continue  # external import; nothing in-scope to resolve to
        target = toplevel_by_file.get(ri.resolved_path, {})
        fmap = imported_names.setdefault(ri.importer_file, {})
        for name in ri.names:
            if name == "*":
                for n, sid in target.items():
                    fmap.setdefault(n, sid)
            elif name in target:
                fmap.setdefault(name, target[name])

    def resolve(callee_name: str, caller: SymbolRecord) -> Optional[str]:
        local = toplevel_by_file.get(caller.path, {})
        if callee_name in local:
            return local[callee_name]
        imported = imported_names.get(caller.path, {})
        if callee_name in imported:
            return imported[callee_name]
        candidates = methods_by_name.get(callee_name, [])
        if caller.parent_id:
            same_class = [c for c in candidates if by_id[c].parent_id == caller.parent_id]
            if len(same_class) == 1:
                return same_class[0]
        if len(candidates) == 1:
            return candidates[0]
        return None

    return resolve


def call_graph_builder(scoped: ScopedSymbols) -> CallGraph:
    """Resolve each function's raw calls to in-scope callee symbol ids and
    emit directed caller→callee CallEdge records (one per resolved call site)."""
    symbols = scoped.resolved_imports.symbol_table.symbols
    resolve = _build_name_resolver(scoped)

    edges: list[CallEdge] = []
    for sym in symbols:
        if sym.kind not in ("function", "method"):
            continue
        for site in sym.call_sites:
            callee_id = resolve(site.callee_name, sym)
            if callee_id is not None and callee_id != sym.id:
                edges.append(CallEdge(
                    caller_id=sym.id,
                    callee_id=callee_id,
                    lineno=site.lineno,
                ))

    return CallGraph(scoped=scoped, edges=edges)


# ===========================================================================
# Stage 5 — dataflow_extractor
# ===========================================================================

def dataflow_extractor(call_graph: CallGraph) -> DataflowGraph:
    """Forward def-use pass over each function's call sites: when a variable
    bound to one call's result is later used as an argument to another call,
    emit a DataflowEdge linking producer callee to consumer callee."""
    symbols = call_graph.scoped.resolved_imports.symbol_table.symbols

    # Same import/scope-aware resolver as the call graph, so a producer/consumer
    # is identified by real binding rather than a last-wins name collision.
    resolve = _build_name_resolver(call_graph.scoped)

    dataflow_edges: list[DataflowEdge] = []

    for sym in symbols:
        if sym.kind not in ("function", "method"):
            continue
        if not sym.call_sites:
            continue

        # Track: var_name → callee_id that last produced it
        produced_by: dict[str, str] = {}

        for site in sym.call_sites:
            callee_id = resolve(site.callee_name, sym)
            # Check if any arg was produced by a previous call
            for arg in site.arg_names:
                if arg in produced_by and callee_id and produced_by[arg]:
                    dataflow_edges.append(DataflowEdge(
                        within=sym.id,
                        from_callee=produced_by[arg],
                        to_callee=callee_id,
                        var=arg,
                    ))

            # Record what this call produces
            if callee_id:
                for var in site.assigned_to:
                    produced_by[var] = callee_id

    return DataflowGraph(call_graph=call_graph, dataflow_edges=dataflow_edges)


# ===========================================================================
# Stage 6 — skeleton_assembler
# ===========================================================================

def skeleton_assembler(dataflow_graph: DataflowGraph) -> CodeSkeleton:
    """Merge symbols, locations, signatures, scope, call graph, and dataflow
    into one unified CodeSkeleton artifact."""
    call_graph = dataflow_graph.call_graph
    scoped = call_graph.scoped
    resolved_imports_obj = scoped.resolved_imports
    symbol_table = resolved_imports_obj.symbol_table

    return CodeSkeleton(
        dataflow_graph=dataflow_graph,
        symbols=symbol_table.symbols,
        imports=symbol_table.imports,
        resolved=resolved_imports_obj.resolved,
        scope_classes=scoped.scope_classes,
        boundary=scoped.boundary,
        call_edges=call_graph.edges,
        dataflow_edges=dataflow_graph.dataflow_edges,
    )


# ===========================================================================
# Top-level entry point
# ===========================================================================

def build_skeleton(manifest: list[FileRecord], source_root: str) -> CodeSkeleton:
    """Run all six stages in order and return the assembled CodeSkeleton."""
    st = symbol_extractor(manifest)
    ri = import_resolver(st, source_root)
    ss = scope_classifier(ri, source_root)
    cg = call_graph_builder(ss)
    dg = dataflow_extractor(cg)
    return skeleton_assembler(dg)
