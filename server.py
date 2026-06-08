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

Execution protocol (both authoring and editing):
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
        "locations": [_loc_dict(loc) for loc in c.locations],
    }


def _warn_dict(w) -> dict:
    return {
        "id": w.id,
        "warning_type": w.warning_type,
        "message": w.message,
        "affected": w.affected,
        "ignored": w.ignored,
        "ignore_reason": w.ignore_reason,
    }


def _active_summary() -> list[dict]:
    """Active (non-ignored) warnings as compact {id, type, affected} dicts, for
    the VERIFY step after a write."""
    return [
        {"id": w.id, "warning_type": w.warning_type, "affected": w.affected}
        for w in _active_warnings(GRAPH.warnings)
    ]


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
    return {
        "name": ACTIVE,
        "graph_path": _active_path(),
        "project_root": _project_root(),
        "components": len(GRAPH.components),
        "edges": len(GRAPH.edges),
        "max_z_level": max_z,
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
    """All current non-ignored warnings, recomputed live against the graph. Each
    warning explains why it fired AND when it is safe to ignore. Warning types:
      hanging_output   -- an output type no downstream component consumes.
      starved_input    -- an input type no upstream component produces.
      undefined_types  -- a non-external component with empty input or output types.
      orphaned_component -- a component with no edges at all.
      coverage_gap     -- a parent's types not covered by its entry/exit children.
      flow_cycle       -- a cycle in the FLOW graph (may be a legit feedback loop).
    Use ignore_warning to dismiss any you have judged intentional.

    Returns {"warnings": [...]}."""
    if GRAPH is None:
        return _no_active()
    _persist()
    return {"warnings": [_warn_dict(w) for w in _active_warnings(GRAPH.warnings)]}


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
    returns {"ok": True, "active_warnings": [...]} -- review those to VERIFY."""
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
    return {"ok": True, "component_id": component_id, "active_warnings": _active_summary()}


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
    primitive). Returns {"ok": True, "active_warnings": [...]} or
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
    return {"ok": True, "component_id": component_id, "active_warnings": _active_summary()}


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
    return {"ok": True, "deleted": component_id, "active_warnings": _active_summary()}


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

    Returns {"ok": True, "active_warnings": [...]} or {"ok": False, "errors": [...]}."""
    if GRAPH is None:
        return _no_active()
    et, err = _parse_edge_type(edge_type)
    if err:
        return err
    errors = _propose_edge(GRAPH, Edge(et, from_id, to_id))
    if errors:
        return {"ok": False, "errors": errors}
    _persist()
    return {"ok": True, "edge": f"{from_id} -> {to_id} ({et.value})", "active_warnings": _active_summary()}


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


if __name__ == "__main__":
    mcp.run()
