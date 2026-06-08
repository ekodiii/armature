import ast
import re
from typing import Optional

from .models import DispatchedFiles, FileRecord, RawNode, StructureMap


# ---------------------------------------------------------------------------
# sa_file_dispatcher
# ---------------------------------------------------------------------------

def sa_file_dispatcher(manifest: list[FileRecord]) -> DispatchedFiles:
    """Partition FileManifest by language so each parser only sees its files."""
    return DispatchedFiles(
        python=[f for f in manifest if f.language == "python"],
        generic=[f for f in manifest if f.language != "python"],
    )


# ---------------------------------------------------------------------------
# sa_ast_extractor — Python side
# ---------------------------------------------------------------------------

def _params_from_args(args: ast.arguments) -> list[str]:
    return [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]


def _return_hint(node) -> Optional[str]:
    if node.returns is None:
        return None
    try:
        return ast.unparse(node.returns)
    except Exception:
        return None


def _extract_imports(tree: ast.AST) -> list[str]:
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}" if module else alias.name)
    return imports


def _extract_calls(node: ast.AST) -> list[str]:
    calls = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                calls.append(func.id)
            elif isinstance(func, ast.Attribute):
                calls.append(func.attr)
    return calls


def _extract_python_nodes(file: FileRecord) -> list[RawNode]:
    try:
        tree = ast.parse(file.text, filename=file.relative_path)
    except SyntaxError:
        return []

    raw_imports = _extract_imports(tree)
    lines = file.text.splitlines()
    nodes: list[RawNode] = []

    # Module-level node
    nodes.append(RawNode(
        node_id=file.relative_path,
        kind="module",
        params=[],
        return_hint=None,
        start_line=1,
        end_line=len(lines),
        relative_path=file.relative_path,
        raw_imports=raw_imports,
        raw_calls=[],
    ))

    for top in ast.iter_child_nodes(tree):
        if isinstance(top, ast.ClassDef):
            class_id = f"{file.relative_path}::{top.name}"
            nodes.append(RawNode(
                node_id=class_id,
                kind="class",
                params=[],
                return_hint=None,
                start_line=top.lineno,
                end_line=getattr(top, "end_lineno", top.lineno),
                relative_path=file.relative_path,
                raw_imports=raw_imports,
                raw_calls=_extract_calls(top),
            ))
            for member in ast.iter_child_nodes(top):
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_id = f"{file.relative_path}::{top.name}::{member.name}"
                    nodes.append(RawNode(
                        node_id=method_id,
                        kind="function",
                        params=_params_from_args(member.args),
                        return_hint=_return_hint(member),
                        start_line=member.lineno,
                        end_line=getattr(member, "end_lineno", member.lineno),
                        relative_path=file.relative_path,
                        raw_imports=raw_imports,
                        raw_calls=_extract_calls(member),
                    ))

        elif isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_id = f"{file.relative_path}::{top.name}"
            nodes.append(RawNode(
                node_id=func_id,
                kind="function",
                params=_params_from_args(top.args),
                return_hint=_return_hint(top),
                start_line=top.lineno,
                end_line=getattr(top, "end_lineno", top.lineno),
                relative_path=file.relative_path,
                raw_imports=raw_imports,
                raw_calls=_extract_calls(top),
            ))

    return nodes


# ---------------------------------------------------------------------------
# sa_ast_extractor — generic (regex-based; tree-sitter is optional)
# ---------------------------------------------------------------------------

def _generic_imports(text: str, language: str) -> list[str]:
    imports = []
    if language in ("typescript", "javascript"):
        for m in re.finditer(
            r'(?:import\s+.*?\s+from\s+["\'](.+?)["\']|require\s*\(\s*["\'](.+?)["\']\s*\))',
            text,
        ):
            imports.append(m.group(1) or m.group(2))
    elif language == "rust":
        for m in re.finditer(r'\buse\s+([\w:]+)', text):
            imports.append(m.group(1))
    elif language == "go":
        for m in re.finditer(r'"([\w./]+)"', text):
            imports.append(m.group(1))
    return imports


def _function_patterns(language: str) -> list[tuple[str, str]]:
    if language in ("typescript", "javascript"):
        return [
            (r'^(?:export\s+)?(?:async\s+)?function\s+(\w+)', "function"),
            (r'^(?:export\s+)?(?:abstract\s+)?class\s+(\w+)', "class"),
            (r'^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(', "function"),
        ]
    elif language == "rust":
        return [
            (r'^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(\w+)', "function"),
            (r'^(?:pub(?:\([^)]*\))?\s+)?struct\s+(\w+)', "class"),
            (r'^(?:pub(?:\([^)]*\))?\s+)?impl(?:\s+\w+\s+for)?\s+(\w+)', "class"),
        ]
    elif language == "go":
        return [
            (r'^func\s+(?:\([^)]+\)\s+)?(\w+)', "function"),
            (r'^type\s+(\w+)\s+struct', "class"),
        ]
    return []


def _extract_generic_nodes(file: FileRecord) -> list[RawNode]:
    lines = file.text.splitlines()
    raw_imports = _generic_imports(file.text, file.language)
    nodes: list[RawNode] = [
        RawNode(
            node_id=file.relative_path,
            kind="module",
            params=[],
            return_hint=None,
            start_line=1,
            end_line=len(lines),
            relative_path=file.relative_path,
            raw_imports=raw_imports,
            raw_calls=[],
        )
    ]

    for pattern, kind in _function_patterns(file.language):
        for m in re.finditer(pattern, file.text, re.MULTILINE):
            name = m.group(1)
            lineno = file.text[: m.start()].count("\n") + 1
            nodes.append(RawNode(
                node_id=f"{file.relative_path}::{name}",
                kind=kind,
                params=[],
                return_hint=None,
                start_line=lineno,
                end_line=lineno,
                relative_path=file.relative_path,
                raw_imports=raw_imports,
                raw_calls=[],
            ))

    return nodes


def sa_ast_extractor(dispatched: DispatchedFiles) -> list[RawNode]:
    """Extract RawNodes from Python (ast) and generic (regex) files; normalise both."""
    nodes: list[RawNode] = []
    for f in dispatched.python:
        nodes.extend(_extract_python_nodes(f))
    for f in dispatched.generic:
        nodes.extend(_extract_generic_nodes(f))
    return nodes


# ---------------------------------------------------------------------------
# sa_dependency_resolver
# ---------------------------------------------------------------------------

def sa_dependency_resolver(raw_nodes: list[RawNode]) -> list[RawNode]:
    """Resolve raw_imports → node_ids (or external:X) and raw_calls → node_ids (or unresolved:X)."""
    path_to_module: dict[str, str] = {}
    name_to_id: dict[str, str] = {}

    for n in raw_nodes:
        if n.kind == "module":
            path_to_module[n.relative_path] = n.node_id
        # leaf name → node_id for call resolution (last segment)
        short = n.node_id.split("::")[-1]
        name_to_id[short] = n.node_id

    for node in raw_nodes:
        resolved_imports: list[str] = []
        for imp in node.raw_imports:
            as_path = imp.replace(".", "/") + ".py"
            if as_path in path_to_module:
                resolved_imports.append(path_to_module[as_path])
            else:
                match = next((p for p in path_to_module if p.endswith(as_path)), None)
                resolved_imports.append(path_to_module[match] if match else f"external:{imp}")
        node.imports = resolved_imports

        resolved_calls: list[str] = []
        for call in node.raw_calls:
            resolved_calls.append(name_to_id.get(call, f"unresolved:{call}"))
        node.calls = resolved_calls

    return raw_nodes


# ---------------------------------------------------------------------------
# sa_external_detector
# ---------------------------------------------------------------------------

_STDLIB_PREFIXES = {
    "os", "sys", "re", "ast", "json", "pathlib", "typing", "dataclasses",
    "collections", "itertools", "functools", "abc", "io", "time", "datetime",
    "subprocess", "threading", "multiprocessing", "logging", "unittest",
    "contextlib", "copy", "math", "hashlib", "base64", "struct", "socket",
    "http", "urllib", "email", "html", "xml", "csv", "sqlite3", "pickle",
    "enum", "inspect", "importlib", "types", "warnings", "traceback",
    "string", "textwrap", "shutil", "tempfile", "glob", "fnmatch",
    "argparse", "configparser", "signal", "gc", "weakref", "array", "queue",
}


def sa_external_detector(nodes: list[RawNode]) -> list[RawNode]:
    """Flag nodes as external=True; create stubs for dangling unresolved call targets."""
    internal_ids = {n.node_id for n in nodes}

    for node in nodes:
        if node.external:
            continue
        all_imports_external = bool(node.imports) and all(
            imp.startswith("external:") for imp in node.imports
        )
        no_internal_callers = not any(c in internal_ids for c in node.calls)
        if all_imports_external and no_internal_callers:
            first_pkg = node.imports[0].replace("external:", "").split(".")[0]
            if first_pkg in _STDLIB_PREFIXES:
                node.external = True

    # Create stub nodes so the call graph has no dangling edges
    stubs: list[RawNode] = []
    existing_ids = {n.node_id for n in nodes}
    for node in nodes:
        for call in node.calls:
            if call.startswith("unresolved:"):
                name = call.removeprefix("unresolved:")
                stub_id = f"__external_stub__::{name}"
                if stub_id not in existing_ids:
                    existing_ids.add(stub_id)
                    stubs.append(RawNode(
                        node_id=stub_id,
                        kind="function",
                        params=[],
                        return_hint=None,
                        start_line=0,
                        end_line=0,
                        relative_path="",
                        raw_imports=[],
                        raw_calls=[],
                        external=True,
                    ))
                # rewrite the unresolved call to point at the stub
                idx = node.calls.index(call)
                node.calls[idx] = stub_id

    return nodes + stubs


# ---------------------------------------------------------------------------
# sa_location_mapper
# ---------------------------------------------------------------------------

def sa_location_mapper(nodes: list[RawNode]) -> StructureMap:
    """Attach FileLocations and assemble the final StructureMap."""
    node_ids = {n.node_id for n in nodes}
    call_edges: list[tuple[str, str]] = []
    import_edges: list[tuple[str, str]] = []

    for node in nodes:
        for call_id in node.calls:
            if call_id in node_ids and call_id != node.node_id:
                call_edges.append((node.node_id, call_id))
        for imp in node.imports:
            if imp.startswith("external:"):
                import_edges.append((node.node_id, imp))
            elif imp in node_ids and imp != node.node_id:
                import_edges.append((node.node_id, imp))

    return StructureMap(nodes=nodes, call_edges=call_edges, import_edges=import_edges)


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def analyze(manifest: list[FileRecord]) -> StructureMap:
    dispatched = sa_file_dispatcher(manifest)
    raw_nodes = sa_ast_extractor(dispatched)
    resolved = sa_dependency_resolver(raw_nodes)
    tagged = sa_external_detector(resolved)
    return sa_location_mapper(tagged)
