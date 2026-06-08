"""Armature MCP server.

Exposes a graph as a set of navigation and authoring tools so an LLM can plan and
edit large engineering systems of any kind (software, mechanical, electrical,
process, organizational) without holding the whole system in context. The LLM
fetches one zoom level at a time; the graph holds all memory between context wipes.

Workspace (how the server finds your work):
- You may keep many named graphs, one per system you are modeling. They live in a
  central library (`~/.armature/graphs/`) by default, so this works the same on a
  terminal, a desktop app, or the web -- there is no dependency on a "current
  directory". A registry maps each graph NAME to its file, so a graph file may also
  live inside a code repo (pass a `path` to new_graph) and still be opened by name.
- Exactly one graph is ACTIVE per session; every read/write tool operates on it.
  The active graph is remembered across restarts.
- Start by calling list_graphs() to see what exists, then open_graph(name) to
  resume one or new_graph(name) to begin a new system. If you are unsure what is
  open, call get_graph_stats().

Graph model (read this before using any tool):
- A component has one input port and one output port, each a list of string
  TYPES (convention, not a registry). It also has a `processing` description, a
  `z_level` (abstraction depth; top level is 0), and an `external` flag.
- Components decompose fractally. A component at z=N is decomposed into children
  at z=N+1. Any child is itself a valid component.
- THREE edge types:
    FLOW      -- data moving between SIBLINGS at the same z-level. No diagonals:
                 both ends must share the same parent.
    SCOPE     -- parent to direct child, exactly one z-level down. You never
                 create these by hand; they are materialized automatically when
                 you propose a component with a parent_id.
    REFERENCE -- a weak, optional annotation link. The ONLY edge allowed to cross
                 parents and z-levels. Use it to record that a deep child feeds
                 another deep child across a decomposition boundary. It is ignored
                 by all contracts (coverage, cycles, FLOW paths) and surfaced only
                 by get_references.
- Validation is non-blocking where it can be: structural errors (bad z-levels,
  non-sibling FLOW, missing parents) block a write; everything else surfaces as a
  WARNING you can read and, if intentional, ignore. Ignored warnings stay ignored
  across sessions.

Execution protocol (all modes):
  1. ORIENT  -- fetch the relevant component / subgraph before touching anything.
  2. PLAN    -- reason about what changes.
  3. WRITE   -- propose one component at a time.
  4. VERIFY  -- re-fetch what you wrote; check edges and active warnings are clean.
  5. REPEAT.
"""

import json
import os
import re
from typing import Optional

from mcp.server.fastmcp import FastMCP

from models import Component, Edge, EdgeType, FileLocation, Graph
from operations import (
    get_impact as _impact,
    get_neighbors as _neighbors,
    get_path as _path,
    get_references as _references,
    get_subgraph as _subgraph,
)
from serializer import load_graph, save_graph
from store import get_component as _get_component
from graph_warnings import (
    get_active_warnings as _active_warnings,
    ignore_warning as _ignore_warning,
    run_all_warnings,
)
from writer import (
    delete_component as _delete_component,
    mark_implemented as _mark_implemented,
    propose_component as _propose_component,
    propose_edge as _propose_edge,
    update_component as _update_component,
)

# --- workspace configuration -----------------------------------------------
# Central library, location-independent so it works on CLI, desktop, and web.
ARMATURE_HOME = os.environ.get("ARMATURE_HOME", os.path.expanduser("~/.armature"))
GRAPHS_DIR = os.path.join(ARMATURE_HOME, "graphs")
REGISTRY_PATH = os.path.join(ARMATURE_HOME, "registry.json")
# Optional override for resolving relative FileLocation paths (get_component_code).
# When unset, paths resolve against the directory holding the active graph file.
PROJECT_ROOT_OVERRIDE = os.environ.get("ARMATURE_PROJECT_ROOT")

os.makedirs(GRAPHS_DIR, exist_ok=True)

# Active-session state.
GRAPH: Optional[Graph] = None
ACTIVE: Optional[str] = None

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _-]*")

mcp = FastMCP("armature", instructions=__doc__)


# --- registry / workspace helpers ------------------------------------------
def _load_registry() -> dict:
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {"active": None, "graphs": {}}


def _save_registry(reg: dict) -> None:
    with open(REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)


def _valid_name(name: str) -> bool:
    return bool(name) and _NAME_RE.fullmatch(name) is not None


def _active_path() -> Optional[str]:
    return _load_registry()["graphs"].get(ACTIVE)


def _project_root() -> str:
    if PROJECT_ROOT_OVERRIDE:
        return PROJECT_ROOT_OVERRIDE
    path = _active_path()
    return os.path.dirname(os.path.abspath(path)) if path else os.getcwd()


def _set_active(name: str, graph: Graph) -> None:
    global GRAPH, ACTIVE
    GRAPH, ACTIVE = graph, name
    reg = _load_registry()
    reg["active"] = name
    _save_registry(reg)


def _no_active() -> dict:
    names = list(_load_registry()["graphs"])
    return {
        "error": (
            "No graph is open. Call open_graph(name) to resume one or new_graph(name) "
            f"to start a new system. Existing graphs: {names or 'none yet'}."
        )
    }


# Resume the last-active graph from a previous session, if still present.
_boot = _load_registry()
if _boot.get("active") and _boot["active"] in _boot["graphs"]:
    _boot_path = _boot["graphs"][_boot["active"]]
    if os.path.exists(_boot_path):
        GRAPH = load_graph(_boot_path)
        ACTIVE = _boot["active"]


# --- internal helpers ------------------------------------------------------
def _persist() -> None:
    """Recompute warnings (preserving ignored state) and write the active graph to
    its registered file. Called after every successful mutation so the graph is the
    durable source of truth between context wipes."""
    GRAPH.warnings = run_all_warnings(GRAPH)
    save_graph(GRAPH, _active_path())


def _fetch(component_id: str):
    try:
        return _get_component(GRAPH, component_id), None
    except KeyError:
        return None, {"error": f"Component '{component_id}' not found."}


def _parse_edge_type(raw: str):
    try:
        return EdgeType(raw.upper()), None
    except ValueError:
        return None, {
            "error": f"Invalid edge_type '{raw}'. Use one of: FLOW, SCOPE, REFERENCE."
        }


def _loc_dict(loc: FileLocation) -> dict:
    return {"path": loc.path, "start_line": loc.start_line, "end_line": loc.end_line}


def _status(c: Component) -> str:
    """Implementation status, derived from version vs implemented_version:
    'planned' (no code yet), 'implemented' (code matches spec), or 'stale' (spec
    was edited after the code was written, so the code needs updating)."""
    if c.implemented_version is None:
        return "planned"
    if c.implemented_version == c.version:
        return "implemented"
    return "stale"


def _comp_dict(c: Component) -> dict:
    return {
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
        "status": _status(c),
        "locations": [_loc_dict(loc) for loc in c.locations],
    }


# One-line meaning + ignore guidance per warning type. Returned once per type that
# is present, instead of repeating the full prose message on every warning.
WARNING_LEGEND = {
    "hanging_output": "An output type has no downstream consumer. OK if it is a terminal/log/side-effect, or wire it to a storage component.",
    "starved_input": "An input type has no upstream producer. OK if it enters at the graph boundary.",
    "undefined_types": "A non-external component has empty input or output types. OK if deliberately abstract at this stage.",
    "orphaned_component": "A component has no edges at all. OK if newly added and not yet wired in.",
    "coverage_gap": "A parent's declared types are not covered by its entry/exit children. OK mid-authoring; resolve before treating the level as done.",
    "flow_cycle": "A cycle exists in the FLOW graph. OK if it is an intentional feedback/retry loop.",
}


def _active_count() -> int:
    """Number of active (non-ignored) warnings. Returned as a cheap signal after a
    write so the caller can decide whether to spend a get_active_warnings call."""
    return len(_active_warnings(GRAPH.warnings))


# ===========================================================================
# Workspace / graph tools
# ===========================================================================
@mcp.tool()
def list_graphs() -> dict:
    """List every graph in the workspace by name, with its file path, component
    count, and which one is currently active. Call this first when you do not know
    what systems already exist -- to resume one (open_graph) or avoid clobbering it.

    Returns {"graphs": [{name, path, active, components, exists}], "active": name}."""
    reg = _load_registry()
    graphs = []
    for name, path in reg["graphs"].items():
        info = {"name": name, "path": path, "active": name == ACTIVE, "exists": os.path.exists(path)}
        if info["exists"]:
            try:
                info["components"] = len(load_graph(path).components)
            except Exception:
                info["exists"] = False
        graphs.append(info)
    return {"graphs": graphs, "active": ACTIVE}


@mcp.tool()
def new_graph(name: str, path: Optional[str] = None, overwrite: bool = False) -> dict:
    """Create a fresh, empty graph, register it under `name`, and make it active.
    This is the entry point for AUTHORING MODE (modeling a system from scratch).

    - name: a human label for the system being modeled (letters, digits, spaces,
      dashes, underscores), e.g. "payment-service" or "HVAC redesign".
    - path: optional. By default the graph is stored centrally in the workspace
      library. Pass a path (e.g. "./armature_graph.yaml") to store the file inside
      a code repo for version control -- it is still opened later by name.
    - overwrite: refuses if `name` is already taken or a file already exists at the
      target, to avoid destroying work. Pass True only with explicit confirmation.
      If the system already exists and you mean to change it, you are in EDITING
      MODE: call open_graph(name) instead, then edit.

    After this, define ALL z=0 components and their FLOW edges before decomposing.
    Work top-down, one level at a time. Returns {"ok": True, "name", "path"} or
    {"error": ...}."""
    if not _valid_name(name):
        return {"error": f"Invalid graph name '{name}'. Use letters, digits, spaces, '-' or '_'."}
    reg = _load_registry()
    target = os.path.abspath(path) if path else os.path.join(GRAPHS_DIR, f"{name}.yaml")
    if (name in reg["graphs"] or os.path.exists(target)) and not overwrite:
        return {
            "error": (
                f"A graph named '{name}' already exists. To resume it call "
                f"open_graph('{name}'); to discard and recreate it pass overwrite=True."
            )
        }
    graph = Graph.new()
    save_graph(graph, target)
    reg["graphs"][name] = target
    _save_registry(reg)
    _set_active(name, graph)
    return {
        "ok": True,
        "name": name,
        "path": target,
        "message": "Empty graph created and active. Define all z=0 components and their FLOW edges before decomposing.",
    }


@mcp.tool()
def open_graph(name: str) -> dict:
    """Make an existing graph active so you can read or edit it (EDITING MODE).
    Use list_graphs() to see available names. Returns the same orientation summary
    as get_graph_stats, or {"error": ...} if the name is unknown or its file is
    missing."""
    reg = _load_registry()
    if name not in reg["graphs"]:
        return {"error": f"No graph named '{name}'. Available: {list(reg['graphs']) or 'none yet'}."}
    path = reg["graphs"][name]
    if not os.path.exists(path):
        return {"error": f"Graph '{name}' is registered but its file is missing at {path}."}
    _set_active(name, load_graph(path))
    return get_graph_stats()


@mcp.tool()
def get_graph_stats() -> dict:
    """Orientation tool for the ACTIVE graph: its name, file path, project root,
    component/edge counts, max z-level reached, and active (non-ignored) warning
    count. Call this when unsure what is open or how far authoring has progressed.

    Returns {"error": ...} with the list of available graphs if none is open."""
    if GRAPH is None:
        return _no_active()
    GRAPH.warnings = run_all_warnings(GRAPH)
    save_graph(GRAPH, _active_path())
    max_z = max((c.z_level for c in GRAPH.components.values()), default=0)
    by_status = {"planned": 0, "implemented": 0, "stale": 0}
    for c in GRAPH.components.values():
        by_status[_status(c)] += 1
    return {
        "name": ACTIVE,
        "graph_path": _active_path(),
        "project_root": _project_root(),
        "components": len(GRAPH.components),
        "edges": len(GRAPH.edges),
        "max_z_level": max_z,
        "by_status": by_status,
        "active_warnings": len(_active_warnings(GRAPH.warnings)),
    }


# ===========================================================================
# Read tools
# ===========================================================================
@mcp.tool()
def get_component(component_id: str) -> dict:
    """Fetch a single component by id, including its ports (input_types,
    output_types), processing description, z_level, external flag, and its
    connection ids (edges_in, edges_out, children, parent_id). This is your
    primary ORIENT tool -- fetch the component you are about to decompose or edit
    before touching anything.

    To resolve edge ids into neighbor components, use get_neighbors. Returns the
    component dict or {"error": ...}."""
    if GRAPH is None:
        return _no_active()
    c, err = _fetch(component_id)
    return err or _comp_dict(c)


@mcp.tool()
def get_neighbors(component_id: str, edge_type: str, upstream: bool = False) -> dict:
    """Components directly connected to this one by edges of a given type.

    edge_type: "FLOW" (siblings exchanging data), "SCOPE" (parent/children), or
               "REFERENCE" (weak cross-boundary links).
    upstream:  False (default) follows outgoing edges -- downstream consumers /
               children. True follows incoming edges -- upstream producers / parent.

    Returns {"neighbors": [component, ...]} or {"error": ...}."""
    if GRAPH is None:
        return _no_active()
    et, err = _parse_edge_type(edge_type)
    if err:
        return err
    _, err = _fetch(component_id)
    if err:
        return err
    return {"neighbors": [_comp_dict(c) for c in _neighbors(GRAPH, component_id, et, upstream)]}


@mcp.tool()
def get_references(component_id: str, incoming: bool = False) -> dict:
    """Weak REFERENCE links for a component, kept separate from FLOW/SCOPE.

    REFERENCE edges record cross-boundary data flow (e.g. a deep child of A feeds
    a deep child of B, where A->B exists one level up). They are pure navigation
    hints and are invisible to coverage, cycle, and path logic.

    incoming=False returns components this one references; incoming=True returns
    components that reference this one. Returns {"references": [...]} or {"error"}."""
    if GRAPH is None:
        return _no_active()
    _, err = _fetch(component_id)
    if err:
        return err
    return {"references": [_comp_dict(c) for c in _references(GRAPH, component_id, incoming)]}


@mcp.tool()
def search_components(query: str) -> dict:
    """Find components whose id, description, or processing text contains the query
    (case-insensitive substring match -- semantic/embedding search is not yet
    wired). Useful when you know roughly what a component does but not its id.

    Returns {"matches": [component, ...]}."""
    if GRAPH is None:
        return _no_active()
    q = query.lower()
    matches = [
        c
        for c in GRAPH.components.values()
        if q in c.component_id.lower()
        or q in c.description.lower()
        or q in c.processing.lower()
    ]
    return {"matches": [_comp_dict(c) for c in matches]}


@mcp.tool()
def get_subgraph(component_id: str, depth: int = -1) -> dict:
    """The SCOPE subtree rooted at a component: the component plus its descendants
    down to `depth` levels (depth=-1, the default, means all the way down; depth=1
    means just the direct children). This is how you zoom IN on one part of the
    system without loading siblings or ancestors.

    Returns {"components": [...]} in pre-order, or {"error": ...}."""
    if GRAPH is None:
        return _no_active()
    _, err = _fetch(component_id)
    if err:
        return err
    return {"components": [_comp_dict(c) for c in _subgraph(GRAPH, component_id, depth)]}


@mcp.tool()
def get_path(from_id: str, to_id: str) -> dict:
    """Shortest FLOW path (list of component ids) from one component to another,
    following data movement at the same z-level. Returns {"path": [...]} or
    {"path": None} if no FLOW path connects them. REFERENCE and SCOPE edges are
    not traversed."""
    if GRAPH is None:
        return _no_active()
    for cid in (from_id, to_id):
        _, err = _fetch(cid)
        if err:
            return err
    return {"path": _path(GRAPH, from_id, to_id)}


@mcp.tool()
def get_impact(component_id: str) -> dict:
    """Everything reachable from a component along FLOW edges, in both directions.
    Use this before editing to understand blast radius: who feeds this component
    (upstream) and who depends on its output (downstream).

    Returns {"upstream": [...], "downstream": [...]} or {"error": ...}."""
    if GRAPH is None:
        return _no_active()
    _, err = _fetch(component_id)
    if err:
        return err
    result = _impact(GRAPH, component_id)
    return {
        "upstream": [_comp_dict(c) for c in result["upstream"]],
        "downstream": [_comp_dict(c) for c in result["downstream"]],
    }


@mcp.tool()
def get_active_warnings() -> dict:
    """All current non-ignored warnings, recomputed live against the graph.

    Each warning is compact: {id, warning_type, affected}. The meaning and
    ignore-guidance for every type present is returned once in `legend` (keyed by
    type), instead of repeating a prose message on every warning. `affected` lists
    the component ids or type names involved. Use ignore_warning(id, reason) to
    dismiss any you have judged intentional.

    Returns {"warnings": [{id, warning_type, affected}], "legend": {type: meaning}}."""
    if GRAPH is None:
        return _no_active()
    _persist()
    active = _active_warnings(GRAPH.warnings)
    return {
        "warnings": [
            {"id": w.id, "warning_type": w.warning_type, "affected": w.affected}
            for w in active
        ],
        "legend": {
            w.warning_type: WARNING_LEGEND[w.warning_type]
            for w in active
            if w.warning_type in WARNING_LEGEND
        },
    }


@mcp.tool()
def get_component_code(component_id: str) -> dict:
    """Read the actual source for a component from the files in its `locations`.
    Returns the slice [start_line, end_line] (1-indexed, inclusive) for each
    location, or the whole file when no line range is set. Requires locations to
    be set -- high-level components often have none; leaf nodes should.

    Paths resolve relative to the project root. Returns {"locations": [{path,
    start_line, end_line, code}]} or {"error": ...}."""
    if GRAPH is None:
        return _no_active()
    c, err = _fetch(component_id)
    if err:
        return err
    if not c.locations:
        return {"error": f"Component '{component_id}' has no locations set."}
    out = []
    for loc in c.locations:
        abspath = os.path.join(_project_root(), loc.path)
        try:
            with open(abspath) as f:
                lines = f.readlines()
        except OSError as e:
            out.append({"path": loc.path, "error": str(e)})
            continue
        start = loc.start_line or 1
        end = loc.end_line or len(lines)
        out.append(
            {
                "path": loc.path,
                "start_line": start,
                "end_line": end,
                "code": "".join(lines[start - 1 : end]),
            }
        )
    return {"locations": out}


@mcp.tool()
def get_work_context(component_id: str) -> dict:
    """The minimal local context needed to safely write or change ONE component --
    call this IMMEDIATELY BEFORE every propose_component / update_component on a
    component, and before implementing its code. Re-pull it each time rather than
    relying on context from earlier in the session; the graph (and your own prior
    writes) may have moved.

    It is bounded to the component's 1-hop frame -- not the whole graph -- so it
    stays cheap. Returns:
      component     -- this node's own contract (types, processing, status, locations)
      receives      -- {upstream_id: output_types} it consumes via FLOW
      must_produce  -- {downstream_id: input_types} its output must satisfy
      parent        -- the parent contract this node helps cover (or null)
      children      -- {child_id: {input_types, output_types, status}} if decomposed
      references    -- weak REFERENCE links (cross-boundary hints)
      code          -- current source at its locations (or null)
      warnings      -- active warnings on this exact node
    """
    if GRAPH is None:
        return _no_active()
    c, err = _fetch(component_id)
    if err:
        return err
    receives = {
        n.component_id: n.output_types
        for n in _neighbors(GRAPH, component_id, EdgeType.FLOW, upstream=True)
    }
    must_produce = {
        n.component_id: n.input_types
        for n in _neighbors(GRAPH, component_id, EdgeType.FLOW, upstream=False)
    }
    parent = None
    if c.parent_id:
        p = _get_component(GRAPH, c.parent_id)
        parent = {
            "component_id": p.component_id,
            "input_types": p.input_types,
            "output_types": p.output_types,
        }
    children = {}
    for ch in c.children:
        cc = _get_component(GRAPH, ch)
        children[ch] = {
            "input_types": cc.input_types,
            "output_types": cc.output_types,
            "status": _status(cc),
        }
    code = get_component_code(component_id).get("locations") if c.locations else None
    GRAPH.warnings = run_all_warnings(GRAPH)
    warns = [
        {"id": w.id, "warning_type": w.warning_type, "affected": w.affected}
        for w in _active_warnings(GRAPH.warnings)
        if w.id.endswith(f"__{component_id}") or component_id in w.affected
    ]
    return {
        "component": {
            "component_id": c.component_id,
            "description": c.description,
            "processing": c.processing,
            "input_types": c.input_types,
            "output_types": c.output_types,
            "z_level": c.z_level,
            "external": c.external,
            "status": _status(c),
            "version": c.version,
            "locations": [_loc_dict(loc) for loc in c.locations],
        },
        "receives": receives,
        "must_produce": must_produce,
        "parent": parent,
        "children": children,
        "references": [n.component_id for n in _references(GRAPH, component_id)],
        "code": code,
        "warnings": warns,
    }


@mcp.tool()
def get_pending_implementation() -> dict:
    """Components whose code does not match the graph: 'planned' (never built) or
    'stale' (spec edited since the code was last written). This is the work queue
    for IMPLEMENTATION MODE. External (atomic boundary) nodes are excluded -- they
    are not ours to implement.

    Sorted stale-first, then leaves-first, then shallowest. Returns
    {"pending": [{component_id, status, z_level, is_leaf, locations}], "count"}."""
    if GRAPH is None:
        return _no_active()
    pending = []
    for c in GRAPH.components.values():
        st = _status(c)
        if st in ("planned", "stale") and not c.external:
            pending.append(
                {
                    "component_id": c.component_id,
                    "status": st,
                    "z_level": c.z_level,
                    "is_leaf": not c.children,
                    "locations": [_loc_dict(loc) for loc in c.locations],
                }
            )
    pending.sort(key=lambda p: (p["status"] != "stale", not p["is_leaf"], p["z_level"]))
    return {"pending": pending, "count": len(pending)}


# ===========================================================================
# Write tools (all validated before commit; the graph is saved on success)
# ===========================================================================
@mcp.tool()
def propose_component(
    component_id: str,
    description: str,
    processing: str,
    input_types: list[str],
    output_types: list[str],
    z_level: int,
    external: bool = False,
    parent_id: Optional[str] = None,
    locations: Optional[list[dict]] = None,
) -> dict:
    """Add a new component to the graph. Propose ONE at a time, then VERIFY.

    - z_level must be 0 for a root (parent_id=None), or parent.z_level + 1 for a
      child. Passing parent_id automatically materializes the SCOPE edge and adds
      the component to the parent's children -- do not create SCOPE edges by hand.
    - external=True marks an atomic boundary node (db driver, third-party API, UI
      input). External components cannot have children and skip type warnings.
    - input_types / output_types are lists of type-name strings. A single type per
      port is ideal; lists are allowed. Leave abstract early in authoring (a
      warning will note it) and refine as you decompose.
    - locations is an optional list of {"path", "start_line"?, "end_line"?} mapping
      the component to source. Optional for high-level nodes; expected for leaves.

    Structural problems (duplicate id, missing/ invalid parent, wrong z_level)
    block the write and come back as {"ok": False, "errors": [...]}. On success
    returns {"ok": True, "active_warnings": <count>}; call get_active_warnings only
    if that count is nonzero and you want to VERIFY details."""
    if GRAPH is None:
        return _no_active()
    locs = [
        FileLocation(
            path=loc["path"],
            start_line=loc.get("start_line"),
            end_line=loc.get("end_line"),
        )
        for loc in (locations or [])
    ]
    component = Component(
        component_id=component_id,
        description=description,
        processing=processing,
        input_types=input_types,
        output_types=output_types,
        z_level=z_level,
        external=external,
        parent_id=parent_id,
        locations=locs,
    )
    errors = _propose_component(GRAPH, component)
    if errors:
        return {"ok": False, "errors": errors}
    _persist()
    return {"ok": True, "component_id": component_id, "active_warnings": _active_count()}


@mcp.tool()
def update_component(component_id: str, fields: dict) -> dict:
    """Edit fields of an existing component. `fields` is a dict of changes.

    Editable: description, processing, input_types, output_types, external,
    locations, z_level. Structural fields (component_id, edges_in, edges_out,
    children, parent_id, version) are managed by the store and CANNOT be set here
    -- to re-parent or rewire, delete and re-propose. Changing z_level re-validates
    every incident edge and is rejected if it would break a FLOW or SCOPE
    relationship. `locations`, if given, must be a list of {"path", "start_line"?,
    "end_line"?} dicts.

    Each successful update bumps the component's version (optimistic-concurrency
    primitive). Returns {"ok": True, "active_warnings": <count>} or
    {"ok": False, "errors": [...]}."""
    if GRAPH is None:
        return _no_active()
    _, err = _fetch(component_id)
    if err:
        return err
    if "locations" in fields:
        fields = dict(fields)
        fields["locations"] = [
            FileLocation(
                path=loc["path"],
                start_line=loc.get("start_line"),
                end_line=loc.get("end_line"),
            )
            for loc in fields["locations"]
        ]
    errors = _update_component(GRAPH, component_id, fields)
    if errors:
        return {"ok": False, "errors": errors}
    _persist()
    return {"ok": True, "component_id": component_id, "active_warnings": _active_count()}


@mcp.tool()
def delete_component(component_id: str) -> dict:
    """Remove a component and all edges touching it (FLOW, SCOPE, REFERENCE).

    Refuses if the component still has children, to avoid orphaning a subtree --
    delete or re-parent the children first. Returns {"ok": True} or
    {"ok": False, "errors": [...]}."""
    if GRAPH is None:
        return _no_active()
    c, err = _fetch(component_id)
    if err:
        return err
    if c.children:
        return {
            "ok": False,
            "errors": [
                f"'{component_id}' still has children {c.children}; delete or re-parent them first."
            ],
        }
    _delete_component(GRAPH, component_id)
    _persist()
    return {"ok": True, "deleted": component_id, "active_warnings": _active_count()}


@mcp.tool()
def propose_edge(edge_type: str, from_id: str, to_id: str) -> dict:
    """Create an edge between two existing components.

    - "FLOW": data from one component to a SIBLING at the same z-level. Both ends
      must share the same parent (top-level roots share parent None). Rejected if
      z-levels differ or the components are not siblings.
    - "REFERENCE": a weak, optional cross-boundary link. May connect any two
      components regardless of parent or z-level. Use it to preserve data-flow
      traceability across a decomposition boundary.
    - "SCOPE": you normally never create these directly -- they are made for you
      when you propose a component with a parent_id. A manual SCOPE edge must point
      from a parent to its direct child exactly one level down.

    Returns {"ok": True, "active_warnings": <count>} or {"ok": False, "errors": [...]}."""
    if GRAPH is None:
        return _no_active()
    et, err = _parse_edge_type(edge_type)
    if err:
        return err
    errors = _propose_edge(GRAPH, Edge(et, from_id, to_id))
    if errors:
        return {"ok": False, "errors": errors}
    _persist()
    return {"ok": True, "edge": f"{from_id} -> {to_id} ({et.value})", "active_warnings": _active_count()}


@mcp.tool()
def ignore_warning(warning_id: str, reason: str) -> dict:
    """Dismiss a warning you have judged intentional (e.g. a hanging_output that is
    genuinely a log side-effect, or a coverage_gap on a level still being authored).
    The ignored state and your reason persist on the graph across sessions, so the
    warning will not resurface. Get warning ids from get_active_warnings.

    Returns {"ok": True} or {"ok": False, "error": ...}."""
    if GRAPH is None:
        return _no_active()
    GRAPH.warnings = run_all_warnings(GRAPH)
    try:
        _ignore_warning(GRAPH.warnings, warning_id, reason)
    except KeyError:
        return {"ok": False, "error": f"Warning '{warning_id}' not found among active warnings."}
    save_graph(GRAPH, _active_path())
    return {"ok": True, "ignored": warning_id}


@mcp.tool()
def mark_implemented(component_id: str) -> dict:
    """Record that this component's code has been written to match its CURRENT spec
    version. Use it in implementation mode after you have written/updated the code
    and set its locations. It advances the component from 'planned'/'stale' to
    'implemented' without changing the spec. Any later edit to the component bumps
    its version and it reads as 'stale' again.

    Returns {"ok": True, "component_id", "status"} or {"error": ...}."""
    if GRAPH is None:
        return _no_active()
    c, err = _fetch(component_id)
    if err:
        return err
    _mark_implemented(GRAPH, component_id)
    save_graph(GRAPH, _active_path())
    return {"ok": True, "component_id": component_id, "status": _status(c)}


# ===========================================================================
# Mode prompts -- surface as slash commands; load the right discipline per task
# ===========================================================================
_WRITE_DISCIPLINE = (
    "Discipline for every write: before you propose or edit ANY component, call "
    "get_work_context(id) for that exact component and read it. Never write from "
    "memory or from context fetched earlier in the session -- the graph changes, "
    "including from your own prior writes. Orient, write, verify, one component at a time."
)


@mcp.prompt()
def armature_plan() -> str:
    """Planning mode: model a new system or change an existing one as a graph."""
    return (
        "You are doing PLANNING WORK in Armature. The graph is the design artifact.\n\n"
        "Orient first:\n"
        "  list_graphs() to see what exists. open_graph(name) to resume one, or\n"
        "  new_graph(name) to begin a new system. If unsure what is open, call get_graph_stats().\n\n"
        "Read the request, then take the matching path:\n\n"
        "  NEW SYSTEM -- no graph exists yet:\n"
        "    Define ALL z=0 components and their FLOW edges before decomposing anything.\n"
        "    Do not go deeper until z=0 is complete and warnings are understood.\n"
        "    Work top-down, one level at a time.\n\n"
        "  CHANGE TO EXISTING SYSTEM:\n"
        "    search_components to find what the request refers to.\n"
        "    get_impact before touching anything -- understand the blast radius first.\n"
        "    Make ONLY changes within the scope of the task.\n"
        "    Editing a spec bumps version and flips status to 'stale' -- surface it, do not hide it.\n\n"
        "  DECOMPOSING A COMPONENT:\n"
        "    get_work_context(id) for the component you are entering.\n"
        "    Work only within its contract. Treat the rest of the graph as invisible.\n"
        "    Entry children must cover parent input types. Exit children must cover output types.\n\n"
        "After each write: check the returned warning count. Call get_active_warnings if nonzero.\n\n"
        + _WRITE_DISCIPLINE
    )


@mcp.prompt()
def armature_implement() -> str:
    """Implementation mode: realize the graph's spec as code, hardware, config, or any artifact."""
    return (
        "You are IMPLEMENTING in Armature. The graph spec is the acceptance criterion.\n\n"
        "This is human-directed. Respond to what the human tells you -- do not autonomously\n"
        "work through the queue unless explicitly asked.\n\n"
        "For each request:\n"
        "1. search_components to find the component(s) that match. If the match is\n"
        "   high-level, check its children -- work at the level where the task lives.\n"
        "2. get_impact to establish the full scope: what is connected, what must agree.\n"
        "3. For each component in scope: get_work_context(id) -- the contract is the\n"
        "   acceptance criterion for this component.\n"
        "4. Produce or verify the artifact:\n"
        "   - If you can produce it (code, config, document, spec): produce it and verify\n"
        "     it honors the contract.\n"
        "   - If the human produced it (wired hardware, deployed config, enacted a process):\n"
        "     verify their work against the contract with them.\n"
        "5. Before calling mark_implemented(id), confirm ALL contracts in scope are\n"
        "   satisfied: inputs covered, outputs produced, no broken edges.\n\n"
        "If reality does not match the spec -- STOP. Surface the discrepancy. Ask:\n"
        "  - Does the artifact need to change? -> continue implementing.\n"
        "  - Does the graph need to change? -> switch to /armature_plan first, then return.\n"
        "Never silently reconcile a mismatch. The graph and reality must agree.\n\n"
        "get_pending_implementation() shows what is still planned or stale -- call it\n"
        "when the human asks what is left to do.\n\n"
        + _WRITE_DISCIPLINE
    )


if __name__ == "__main__":
    mcp.run()
