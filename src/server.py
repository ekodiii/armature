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

import copy
import dataclasses
import json
import os
import re
import shutil
from typing import Optional

from mcp.server.fastmcp import FastMCP

from models import Component, Edge, EdgeType, FileLocation, Graph
from operations import (
    get_impact as _impact,
    get_neighbors as _neighbors,
    get_path as _path,
    get_references as _references,
    get_subgraph as _subgraph,
    rank as _rank,
)
from serializer import load_graph, save_graph
from gitsync import (
    GitError as _GitError,
    current_head as _current_head,
    reconcile as _reconcile,
)
from store import get_component as _get_component, remove_edge as _remove_edge
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
from translator import lift as _lift
from translator import verify as _verifylib
from verifier import LeafContract as _LeafContract, Property as _Property, verify_leaf as _verify_leaf
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

# Translation-session state: the skeleton is expensive to build, so it is cached
# across translate_* calls (coverage/stitch/verify reuse it) until the next
# translate_prepare rebuilds it for a (possibly different) source tree.
_TRANSLATE_SKELETON = None
_TRANSLATE_ROOT: Optional[str] = None

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
    global GRAPH, ACTIVE, _TRANSLATE_SKELETON, _TRANSLATE_ROOT
    GRAPH, ACTIVE = graph, name
    # The cached skeleton belongs to whatever graph was active when it was built.
    # Switching graphs invalidates it, or translate_* would audit this graph
    # against the previous project's source tree.
    _TRANSLATE_SKELETON = None
    _TRANSLATE_ROOT = None
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


# Resume the last-active graph from a previous session, if still present. A
# corrupt or unreadable file must not crash server startup — the operator can
# still open another graph.
_boot = _load_registry()
if _boot.get("active") and _boot["active"] in _boot["graphs"]:
    _boot_path = _boot["graphs"][_boot["active"]]
    if os.path.exists(_boot_path):
        try:
            GRAPH = load_graph(_boot_path)
            ACTIVE = _boot["active"]
        except Exception:
            GRAPH = None
            ACTIVE = None


# --- internal helpers ------------------------------------------------------
def _recompute_warnings() -> None:
    """Refresh derived warnings in memory (preserving ignored state) without
    touching disk. Used by read paths so a query never dirties the graph file."""
    GRAPH.warnings = run_all_warnings(GRAPH)


def _persist() -> None:
    """Recompute warnings and write the active graph to its registered file.
    Called after every successful mutation so the graph is the durable source of
    truth between context wipes."""
    _recompute_warnings()
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
    """Implementation status: 'planned' (no code yet), 'drifted' (the anchored
    code changed in git since it was last verified — flagged by reconcile),
    'stale' (spec edited after the code was written), or 'implemented' (code
    matches the current spec)."""
    if c.implemented_version is None:
        return "planned"
    if c.code_drifted:
        return "drifted"
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
        **({"feature": c.feature} if c.feature else {}),
    }


def _terse(c: Component, fields: Optional[list[str]] = None) -> dict:
    """Compact view of a component for token-efficient read responses.

    Default (fields=None) emits: id, one-line description, input_types,
    output_types, n_children, n_edges, status.  Pass `fields` with any subset
    of {"processing", "locations", "edges_in", "edges_out", "children",
    "parent_id", "version", "z_level", "external"} to expand specific keys.
    """
    desc = c.description.split("\n")[0]  # first line only
    out: dict = {
        "component_id": c.component_id,
        "description": desc,
        "input_types": c.input_types,
        "output_types": c.output_types,
        "n_children": len(c.children),
        "n_edges": len(c.edges_in) + len(c.edges_out),
        "status": _status(c),
    }
    if fields:
        for f in fields:
            if f == "processing":
                out["processing"] = c.processing
            elif f == "locations":
                out["locations"] = [_loc_dict(loc) for loc in c.locations]
            elif f == "edges_in":
                out["edges_in"] = c.edges_in
            elif f == "edges_out":
                out["edges_out"] = c.edges_out
            elif f == "children":
                out["children"] = c.children
            elif f == "parent_id":
                out["parent_id"] = c.parent_id
            elif f == "version":
                out["version"] = c.version
            elif f == "z_level":
                out["z_level"] = c.z_level
            elif f == "external":
                out["external"] = c.external
    return out


# Format legend returned once per terse response instead of repeating JSON keys
# on every node. Status is omitted for 'implemented' (the common case).
LINE_LEGEND = (
    "id [input_types -> output_types] <children>ch <edges>e (STATUS unless "
    "implemented) :: description | locations as '@ path:start-end; ...'"
)

# Locations shown per line before collapsing to '+N more'.
_LINE_MAX_LOCS = 5


def _line(c: Component, locations: bool = False, indent: int = 0) -> str:
    """One-line text view of a component -- the terse default for every
    list-returning read tool. A formatted line carries the same information as
    the _terse dict at a fraction of the tokens; LINE_LEGEND documents the
    format once per response."""
    desc = c.description.split("\n")[0]
    ins = ",".join(c.input_types) or "-"
    outs = ",".join(c.output_types) or "-"
    parts = [
        "  " * indent + c.component_id,
        f"[{ins} -> {outs}]",
        f"{len(c.children)}ch",
        f"{len(c.edges_in) + len(c.edges_out)}e",
    ]
    st = _status(c)
    if st != "implemented":
        parts.append(f"({st.upper()})")
    if c.feature:
        parts.append(f"{{feature:{c.feature}}}")
    parts.append(f":: {desc}")
    if locations and c.locations:
        shown = [
            loc.path
            + (f":{loc.start_line}-{loc.end_line}" if loc.start_line else "")
            for loc in c.locations[:_LINE_MAX_LOCS]
        ]
        extra = len(c.locations) - _LINE_MAX_LOCS
        if extra > 0:
            shown.append(f"+{extra} more")
        parts.append("@ " + "; ".join(shown))
    return " ".join(parts)


_STUB_DETAIL = {
    "planned": "no code exists yet (stub)",
    "stale": "its spec changed after its code was written",
    "drifted": "its code changed since it was last verified",
}


def _edge_cautions(graph: Graph, from_id: str, to_id: str) -> list[str]:
    """Stub alerts for both endpoints of a just-created edge."""
    return [
        a
        for role, cid in (("from-endpoint", from_id), ("to-endpoint", to_id))
        for a in [_stub_alert(role, _get_component(graph, cid))]
        if a
    ]


def _stub_alert(role: str, c: Component) -> Optional[str]:
    """One-line alert when a node an agent is about to build against is not
    actually implemented. Externals are excluded: they derive status 'planned'
    by construction (nothing of ours to implement), so flagging them is noise.
    Exp 3 showed agents wire into stubs without noticing unless the signal is
    pushed at them -- this is that push."""
    st = _status(c)
    if c.external or st == "implemented":
        return None
    return f"{role} '{c.component_id}' is {st.upper()} -- {_STUB_DETAIL[st]}"


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
    # overwrite is destructive: keep a one-deep backup so an accidental clobber
    # of an existing graph (especially one in the central library, which has no
    # VCS behind it) is recoverable.
    if os.path.exists(target) and overwrite:
        try:
            shutil.copy2(target, target + ".bak")
        except OSError:
            pass
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
    try:
        graph = load_graph(path)
    except Exception as e:
        return {"error": f"Failed to load graph '{name}' from {path}: {e}"}
    _set_active(name, graph)
    return get_graph_stats()


@mcp.tool()
def get_graph_stats() -> dict:
    """Orientation tool for the ACTIVE graph: its name, file path, project root,
    component/edge counts, max z-level reached, and active (non-ignored) warning
    count. Call this when unsure what is open or how far authoring has progressed.

    Counts describe the AS-BUILT base; planned-feature nodes are summarized
    separately under `features`. Returns {"error": ...} if none is open."""
    if GRAPH is None:
        return _no_active()
    _recompute_warnings()
    base = [c for c in GRAPH.components.values() if c.feature is None]
    max_z = max((c.z_level for c in base), default=0)
    by_status = {"planned": 0, "implemented": 0, "stale": 0, "drifted": 0}
    for c in base:
        by_status[_status(c)] += 1
    features: dict[str, int] = {}
    for c in GRAPH.components.values():
        if c.feature is not None:
            features[c.feature] = features.get(c.feature, 0) + 1
    return {
        "name": ACTIVE,
        "graph_path": _active_path(),
        "project_root": _project_root(),
        "components": len(base),
        "edges": len(GRAPH.edges),
        "max_z_level": max_z,
        "by_status": by_status,
        "features": features,
        "last_synced_sha": GRAPH.last_synced_sha,
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
def get_neighbors(component_id: str, edge_type: str, upstream: bool = False, terse: bool = True) -> dict:
    """Components directly connected to this one by edges of a given type.

    edge_type: "FLOW" (siblings exchanging data), "SCOPE" (parent/children), or
               "REFERENCE" (weak cross-boundary links).
    upstream:  False (default) follows outgoing edges -- downstream consumers /
               children. True follows incoming edges -- upstream producers / parent.
    terse:     True (default) returns one compact text line per component (see
               `legend` in the response). Pass False for full component dicts.

    Returns {"neighbors": [component, ...]} or {"error": ...}."""
    if GRAPH is None:
        return _no_active()
    et, err = _parse_edge_type(edge_type)
    if err:
        return err
    _, err = _fetch(component_id)
    if err:
        return err
    fmt = _line if terse else _comp_dict
    out: dict = {"neighbors": [fmt(c) for c in _neighbors(GRAPH, component_id, et, upstream)]}
    if terse:
        out["legend"] = LINE_LEGEND
    return out


@mcp.tool()
def get_references(component_id: str, incoming: bool = False, terse: bool = True) -> dict:
    """Weak REFERENCE links for a component, kept separate from FLOW/SCOPE.

    REFERENCE edges record cross-boundary data flow (e.g. a deep child of A feeds
    a deep child of B, where A->B exists one level up). They are pure navigation
    hints and are invisible to coverage, cycle, and path logic.

    incoming=False returns components this one references; incoming=True returns
    components that reference this one.
    terse=True (default) returns one compact text line per component; pass False
    for full dicts.

    Returns {"references": [...]} or {"error"}."""
    if GRAPH is None:
        return _no_active()
    _, err = _fetch(component_id)
    if err:
        return err
    fmt = _line if terse else _comp_dict
    out: dict = {"references": [fmt(c) for c in _references(GRAPH, component_id, incoming)]}
    if terse:
        out["legend"] = LINE_LEGEND
    return out


@mcp.tool()
def search_components(query: str, terse: bool = True, limit: int = 20) -> dict:
    """Find components whose id, description, or processing text contains the query
    (case-insensitive substring match -- semantic/embedding search is not yet
    wired). Useful when you know roughly what a component does but not its id.

    terse=True (default) returns one compact text line per match; pass False for
    full dicts. At most `limit` matches are returned (default 20); the response
    reports the total when truncated -- narrow the query rather than raising it.

    Returns {"matches": [component, ...], "total"}."""
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
    total = len(matches)
    fmt = _line if terse else _comp_dict
    out: dict = {"matches": [fmt(c) for c in matches[:limit]], "total": total}
    if total > limit:
        out["note"] = f"showing {limit} of {total} matches; narrow the query"
    if terse:
        out["legend"] = LINE_LEGEND
    return out


@mcp.tool()
def get_subgraph(component_id: str, depth: int = -1, terse: bool = True) -> dict:
    """The SCOPE subtree rooted at a component: the component plus its descendants
    down to `depth` levels (depth=-1, the default, means all the way down; depth=1
    means just the direct children). This is how you zoom IN on one part of the
    system without loading siblings or ancestors.

    terse=True (default) returns one compact text line per component; pass False
    for full dicts.

    Returns {"components": [...]} in pre-order, or {"error": ...}."""
    if GRAPH is None:
        return _no_active()
    _, err = _fetch(component_id)
    if err:
        return err
    fmt = _line if terse else _comp_dict
    out: dict = {"components": [fmt(c) for c in _subgraph(GRAPH, component_id, depth)]}
    if terse:
        out["legend"] = LINE_LEGEND
    return out


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
def get_impact(component_id: str, terse: bool = True) -> dict:
    """Everything reachable from a component along FLOW edges, in both directions.
    Use this before editing to understand blast radius: who feeds this component
    (upstream) and who depends on its output (downstream).

    terse=True (default) returns one compact text line per component; pass False
    for full dicts.

    Returns {"upstream": [...], "downstream": [...]} or {"error": ...}."""
    if GRAPH is None:
        return _no_active()
    _, err = _fetch(component_id)
    if err:
        return err
    result = _impact(GRAPH, component_id)
    fmt = _line if terse else _comp_dict
    out: dict = {
        "upstream": [fmt(c) for c in result["upstream"]],
        "downstream": [fmt(c) for c in result["downstream"]],
    }
    if terse:
        out["legend"] = LINE_LEGEND
    return out


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
    _recompute_warnings()
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
def get_orient(feature: Optional[str] = None, depth: int = 1) -> dict:
    """Compact whole-graph orientation map: the SCOPE tree as indented one-line
    text entries (see `legend`) so an operator grasps the layout without crawling
    get_subgraph node by node.

    depth bounds the walk: 1 (default) = roots + their direct children, 0 =
    roots only, -1 = every level. When levels are hidden the response reports
    how many nodes remain below; expand with get_orient(depth=-1) or zoom into
    one subtree with get_subgraph(id).

    By default this shows the AS-BUILT base only — planned-feature nodes are
    hidden so the map reflects the running system. Pass feature=NAME to overlay
    that feature's planned nodes on top of the base (each marked with its
    feature). Use list_features() to see what features exist.

    Returns {"map": [line, ...], "count", "total", "legend"} in pre-order
    (parents before children)."""
    if GRAPH is None:
        return _no_active()

    def in_view(c: Component) -> bool:
        return c.feature is None or c.feature == feature

    roots = [c for c in GRAPH.components.values() if c.parent_id is None and in_view(c)]
    roots.sort(key=lambda c: c.component_id)

    result: list[str] = []

    def walk(cid: str, level: int):
        c = GRAPH.components.get(cid)
        if c is None or not in_view(c):
            return
        if depth != -1 and level > depth:
            return
        result.append(_line(c, indent=level))
        for child_id in sorted(c.children):
            walk(child_id, level + 1)

    for root in roots:
        walk(root.component_id, 0)

    total = sum(1 for c in GRAPH.components.values() if in_view(c))
    out: dict = {
        "map": result,
        "count": len(result),
        "total": total,
        "legend": LINE_LEGEND,
        "feature": feature,
    }
    if total > len(result):
        out["note"] = (
            f"{total - len(result)} deeper node(s) hidden at depth={depth}; "
            "use get_orient(depth=-1) or get_subgraph(id) to expand"
        )
    return out


@mcp.tool()
def locate(query: str, top_k: int = 4) -> dict:
    """Intent-to-location router: given a keyword or question, returns the most
    relevant components ranked by relevance (id/description/processing match plus
    structural proximity), one compact text line each (see `legend`) WITH its
    exact file locations so you go from intent to precise code range in one call.

    Args:
        query:  keyword or short phrase to match against component ids,
                descriptions, and processing text.
        top_k:  maximum number of results to return (default 4).

    Returns {"matches": [line, ...], "legend"} in rank order, or {"error": ...}."""
    if GRAPH is None:
        return _no_active()
    ranked = _rank(GRAPH, query, top_k=top_k)
    matches = []
    for cid, score in ranked:
        c = GRAPH.components.get(cid)
        if c is None:
            continue
        matches.append(_line(c, locations=True))
    return {"matches": matches, "query": query, "legend": LINE_LEGEND}


@mcp.tool()
def get_component_code(component_id: str, max_lines: int = 200) -> dict:
    """Read the actual source for a component from the files in its `locations`.
    Returns the slice [start_line, end_line] (1-indexed, inclusive) for each
    location, or the whole file when no line range is set. Requires locations to
    be set -- high-level components often have none; leaf nodes should.

    Output is budgeted: at most `max_lines` total lines of code (default 200)
    across all locations. A location cut off by the budget carries
    `omitted_lines`, and the response carries a `note` -- either raise max_lines
    or Read the reported path:range directly. Multi-file components list every
    location even when the budget is spent (with code omitted), so you always
    see the full anchor set.

    Paths resolve relative to the project root. Returns {"locations": [{path,
    start_line, end_line, code}]} or {"error": ...}."""
    if GRAPH is None:
        return _no_active()
    c, err = _fetch(component_id)
    if err:
        return err
    if not c.locations:
        return {"error": f"Component '{component_id}' has no locations set."}
    root = os.path.realpath(_project_root())
    out = []
    budget = max_lines
    truncated = False
    for loc in c.locations:
        # Containment check: a graph file can come from anywhere (shared, checked
        # into a repo), so a location path must not escape the project root via
        # "../" or an absolute path. Resolve symlinks before comparing.
        abspath = os.path.realpath(os.path.join(root, loc.path))
        if abspath != root and not abspath.startswith(root + os.sep):
            out.append({"path": loc.path, "error": "path escapes project root"})
            continue
        try:
            with open(abspath) as f:
                lines = f.readlines()
        except OSError as e:
            out.append({"path": loc.path, "error": str(e)})
            continue
        start = loc.start_line or 1
        end = loc.end_line or len(lines)
        entry = {"path": loc.path, "start_line": start, "end_line": end}
        n = end - start + 1
        if n > budget:
            truncated = True
            entry["code"] = "".join(lines[start - 1 : start - 1 + max(budget, 0)])
            entry["omitted_lines"] = n - max(budget, 0)
            budget = 0
        else:
            entry["code"] = "".join(lines[start - 1 : end])
            budget -= n
        out.append(entry)
    result: dict = {"locations": out}
    if truncated:
        result["note"] = (
            f"output capped at max_lines={max_lines}; locations with "
            "omitted_lines were cut off -- raise max_lines or Read the "
            "reported path:range directly"
        )
    return result


@mcp.tool()
def get_work_context(
    component_id: str,
    terse: bool = False,
    expand_fields: Optional[list[str]] = None,
) -> dict:
    """The minimal local context needed to safely write or change ONE component --
    call this IMMEDIATELY BEFORE every propose_component / update_component on a
    component, and before implementing its code. Re-pull it each time rather than
    relying on context from earlier in the session; the graph (and your own prior
    writes) may have moved.

    It is bounded to the component's 1-hop frame -- not the whole graph -- so it
    stays cheap. Returns:
      alerts        -- FIRST key, present only when something in the frame is NOT
                       implemented: this node or a FLOW producer/consumer, child,
                       or reference that is planned (stub), stale, or drifted.
                       Do not build against an alerted node without surfacing it.
      component     -- this node's own contract (types, processing, status, locations)
      receives      -- {upstream_id: output_types} it consumes via FLOW
      must_produce  -- {downstream_id: input_types} its output must satisfy
      parent        -- the parent contract this node helps cover (or null)
      children      -- {child_id: {input_types, output_types, status}} if decomposed
      references    -- weak REFERENCE links (cross-boundary hints)
      code          -- current source at its locations (or null)
      warnings      -- active warnings on this exact node

    Token-efficiency options:
      terse=True         -- route neighbor/child serialization through the compact
                           shaper (id + one-line desc + types + counts); the focal
                           component is always returned in full.
      expand_fields=[..] -- when terse=True, also include these extra keys for each
                           terse neighbor (e.g. ["locations", "processing"]).
    """
    if GRAPH is None:
        return _no_active()
    c, err = _fetch(component_id)
    if err:
        return err
    ups = _neighbors(GRAPH, component_id, EdgeType.FLOW, upstream=True)
    downs = _neighbors(GRAPH, component_id, EdgeType.FLOW, upstream=False)
    kids = [_get_component(GRAPH, ch) for ch in c.children]
    refs = _references(GRAPH, component_id)
    alerts = [
        a
        for a in (
            [_stub_alert("THIS component", c)]
            + [_stub_alert("upstream producer", n) for n in ups]
            + [_stub_alert("downstream consumer", n) for n in downs]
            + [_stub_alert("child", n) for n in kids]
            + [_stub_alert("referenced component", n) for n in refs]
        )
        if a
    ]
    receives = {n.component_id: n.output_types for n in ups}
    must_produce = {n.component_id: n.input_types for n in downs}
    parent = None
    if c.parent_id:
        p = _get_component(GRAPH, c.parent_id)
        parent = {
            "component_id": p.component_id,
            "input_types": p.input_types,
            "output_types": p.output_types,
        }
    children: dict
    if terse:
        children = {cc.component_id: _terse(cc, fields=expand_fields) for cc in kids}
    else:
        children = {
            cc.component_id: {
                "input_types": cc.input_types,
                "output_types": cc.output_types,
                "status": _status(cc),
            }
            for cc in kids
        }
    code = (
        get_component_code(component_id, max_lines=400).get("locations")
        if c.locations
        else None
    )
    _recompute_warnings()
    warns = [
        {"id": w.id, "warning_type": w.warning_type, "affected": w.affected}
        for w in _active_warnings(GRAPH.warnings)
        if w.id.endswith(f"__{component_id}") or component_id in w.affected
    ]
    out: dict = {}
    if alerts:
        out["alerts"] = alerts
    out.update(
        {
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
            "references": [n.component_id for n in refs],
            "code": code,
            "warnings": warns,
        }
    )
    return out


@mcp.tool()
def get_pending_implementation(feature: Optional[str] = None) -> dict:
    """Components whose code does not match the graph: 'planned' (never built),
    'stale' (spec edited since the code was last written), or 'drifted' (code
    changed in git since it was last verified). This is the work queue for
    IMPLEMENTATION MODE. External (atomic boundary) nodes are excluded -- they
    are not ours to implement.

    By default this is the AS-BUILT queue (base nodes only). Pass feature=NAME to
    get the build queue for a planned feature instead (its own nodes).

    Sorted drifted/stale-first, then leaves-first, then shallowest. Returns
    {"pending": [{component_id, status, z_level, is_leaf, locations}], "count"}."""
    if GRAPH is None:
        return _no_active()
    pending = []
    for c in GRAPH.components.values():
        if c.feature != feature:
            continue
        st = _status(c)
        if st in ("planned", "stale", "drifted") and not c.external:
            pending.append(
                {
                    "component_id": c.component_id,
                    "status": st,
                    "z_level": c.z_level,
                    "is_leaf": not c.children,
                    "locations": [_loc_dict(loc) for loc in c.locations],
                }
            )
    pending.sort(key=lambda p: (p["status"] not in ("stale", "drifted"), not p["is_leaf"], p["z_level"]))
    return {"pending": pending, "count": len(pending)}


@mcp.tool()
def list_features() -> dict:
    """List the PLANNED features layered beside the as-built base — proposals that
    do not yet exist in code and are excluded from base views. Each feature is a
    named set of components you author with propose_component(..., feature=NAME)
    and build later; get_orient(feature=NAME) overlays one on the base, and
    get_pending_implementation(feature=NAME) is its build queue.

    Returns {"features": [{name, components, z0_roots}], "count"}."""
    if GRAPH is None:
        return _no_active()
    feats: dict[str, dict] = {}
    for c in GRAPH.components.values():
        if c.feature is None:
            continue
        f = feats.setdefault(c.feature, {"name": c.feature, "components": 0, "z0_roots": []})
        f["components"] += 1
        if c.parent_id is None:
            f["z0_roots"].append(c.component_id)
    return {"features": sorted(feats.values(), key=lambda f: f["name"]), "count": len(feats)}


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
    feature: Optional[str] = None,
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
    - feature: name a PLANNED feature this node belongs to. Feature nodes are a
      proposal layered beside the as-built base; base views (warnings, orient,
      stats, the impl queue) exclude them, so planning never pollutes the live
      graph. Omit it for as-built work. A node joins the base when its code lands.

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
        feature=feature,
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

    If an endpoint is not implemented (planned stub / stale / drifted), the edge
    is still created but the response carries a `caution` -- surface it before
    building code that assumes the other side exists.

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
    out: dict = {"ok": True, "edge": f"{from_id} -> {to_id} ({et.value})", "active_warnings": _active_count()}
    cautions = _edge_cautions(GRAPH, from_id, to_id)
    if cautions:
        out["caution"] = cautions
    return out


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
    and set its locations. It advances the component from 'planned'/'stale'/'drifted'
    to 'implemented' without changing the spec, and stamps the current git commit
    as the baseline for future drift detection. Any later spec edit reads as
    'stale'; any later code change (detected by reconcile) reads as 'drifted'.

    Returns {"ok": True, "component_id", "status"} or {"error": ...}."""
    if GRAPH is None:
        return _no_active()
    c, err = _fetch(component_id)
    if err:
        return err
    _mark_implemented(GRAPH, component_id, sha=_repo_head())
    save_graph(GRAPH, _active_path())
    return {"ok": True, "component_id": component_id, "status": _status(c)}


@mcp.tool()
def reconcile(since: Optional[str] = None, include_uncommitted: bool = False) -> dict:
    """Reconcile the active graph against git: detect components whose anchored
    code has CHANGED since it was last verified, and flag them 'drifted'. This is
    how the graph stays honest when code is edited outside Armature (a normal
    editor, another agent, a teammate's commit) — nothing else flips those nodes.

    Each implemented component is diffed against its own verified baseline commit
    (its implemented_sha), falling back to `since` or the graph's last_synced_sha.
    Components with no baseline are reported so you can re-mark them. Advances
    last_synced_sha to HEAD.

    - since: optional commit to diff from for components lacking their own baseline.
    - include_uncommitted: also count working-tree edits not yet committed.

    Returns {ok, head, drifted:[{component_id, paths}], recovered:[...],
    no_baseline:[...], checked, clean} or {"error": ...}."""
    if GRAPH is None:
        return _no_active()
    try:
        report = _reconcile(GRAPH, _project_root(), since=since,
                            include_uncommitted=include_uncommitted)
    except _GitError as e:
        return {"error": f"git-sync could not run: {e}"}
    _persist()
    return {
        "ok": True,
        "head": report.head,
        "drifted": [{"component_id": cid, "paths": paths} for cid, paths in report.drifted],
        "recovered": report.recovered,
        "no_baseline": report.no_baseline,
        "checked": report.checked,
        "clean": report.is_clean,
    }


@mcp.tool()
def verify_contract(
    component_id: str,
    properties: list[dict],
    strategies: Optional[dict] = None,
    symbol: Optional[str] = None,
    timeout_s: float = 30.0,
) -> dict:
    """Behaviorally verify a leaf against operator-authored properties (beyond
    structural fidelity). YOU author the property statements after reading the
    leaf's contract/code (get_work_context / get_component_code); this tool
    recovers the callable's signature from the code anchor, assembles a
    Hypothesis property test, RUNS it against the real code in an isolated
    subprocess, and returns the verdict. The run is the check — a false property
    comes back 'refuted' with a counterexample; you cannot talk past it.

    - properties: [{"name", "expression", "kind"?}] — each `expression` is a
      Python bool over the leaf's parameter names, `result` (the return value),
      and the leaf symbol itself (for metamorphic relations).
    - strategies: optional {type-name-or-param-name: hypothesis-strategy-expr}
      for parameters the type registry can't map (e.g. un-annotated or custom
      types), such as {"x": "st.integers()"}.
    - symbol: disambiguate which callable when the file has several.

    Returns {ok, component_id, status, detail, counterexample, evidence} where
    status is verified / refuted / error / unverifiable, or {"error": ...}."""
    if GRAPH is None:
        return _no_active()
    c, err = _fetch(component_id)
    if err:
        return err
    if c.children:
        return {"error": f"Component '{component_id}' is not a leaf; verify targets leaf components."}

    props = []
    for i, p in enumerate(properties or []):
        if not isinstance(p, dict) or not p.get("expression"):
            return {"error": f"properties[{i}] must be a dict with a non-empty 'expression'."}
        props.append(_Property(
            name=p.get("name") or f"prop_{i}",
            expression=p["expression"],
            kind=p.get("kind", "postcondition"),
        ))
    if not props:
        return {"error": "verify_contract needs at least one property {name, expression}."}

    leaf = _LeafContract(
        component_id=c.component_id,
        input_types=list(c.input_types),
        output_types=list(c.output_types),
        processing=c.processing,
        locations=[
            {"path": loc.path, "start_line": loc.start_line, "end_line": loc.end_line}
            for loc in c.locations
        ],
        external=c.external,
    )
    result = _verify_leaf(
        leaf, props,
        project_root=_project_root(),
        external=c.external,
        strategies=strategies,
        symbol=symbol,
        timeout_s=timeout_s,
    )
    return {
        "ok": True,
        "component_id": component_id,
        "status": result.status,
        "detail": result.detail,
        "counterexample": result.counterexample,
        "evidence": result.evidence,
    }


def _repo_head() -> Optional[str]:
    """Current git HEAD of the project root, or None if it is not a repo / git is
    unavailable — mark_implemented still works, just without a drift baseline."""
    try:
        return _current_head(_project_root())
    except _GitError:
        return None


# ===========================================================================
# Batch mutation -- validate-then-commit, single persist
# ===========================================================================
def _apply_batch_op(graph: Graph, index: int, op: dict) -> dict:
    """Apply one batch op to `graph`; returns {index, op, ok, id?, error?}.
    Hard errors (unknown id, duplicate, bad z_level) are returned as ok=False.
    Graph-consistency warnings are NOT checked here — only at final commit."""
    verb = op["op"]

    if verb == "propose_component":
        locs = [
            FileLocation(path=l["path"], start_line=l.get("start_line"), end_line=l.get("end_line"))
            for l in (op.get("locations") or [])
        ]
        component = Component(
            component_id=op["component_id"],
            description=op["description"],
            processing=op["processing"],
            input_types=op["input_types"],
            output_types=op["output_types"],
            z_level=op["z_level"],
            external=op.get("external", False),
            parent_id=op.get("parent_id"),
            locations=locs,
            feature=op.get("feature"),
        )
        errors = _propose_component(graph, component)
        if errors:
            return {"index": index, "op": verb, "ok": False, "error": errors[0]}
        return {"index": index, "op": verb, "ok": True, "id": op["component_id"]}

    if verb == "update_component":
        cid = op["component_id"]
        if cid not in graph.components:
            return {"index": index, "op": verb, "ok": False, "error": f"component '{cid}' not found"}
        fields = dict(op["fields"])
        if "locations" in fields:
            fields["locations"] = [
                FileLocation(path=l["path"], start_line=l.get("start_line"), end_line=l.get("end_line"))
                for l in fields["locations"]
            ]
        errors = _update_component(graph, cid, fields)
        if errors:
            return {"index": index, "op": verb, "ok": False, "error": errors[0]}
        return {"index": index, "op": verb, "ok": True, "id": cid}

    if verb == "delete_component":
        cid = op["component_id"]
        if cid not in graph.components:
            return {"index": index, "op": verb, "ok": False, "error": f"component '{cid}' not found"}
        c = graph.components[cid]
        if c.children:
            return {"index": index, "op": verb, "ok": False, "error": f"'{cid}' still has children {c.children}"}
        _delete_component(graph, cid)
        return {"index": index, "op": verb, "ok": True, "id": cid}

    if verb == "propose_edge":
        et, err = _parse_edge_type(op["edge_type"])
        if err:
            return {"index": index, "op": verb, "ok": False, "error": err["error"]}
        errors = _propose_edge(graph, Edge(et, op["from_id"], op["to_id"]))
        if errors:
            return {"index": index, "op": verb, "ok": False, "error": errors[0]}
        result = {"index": index, "op": verb, "ok": True, "id": f"{op['from_id']} -> {op['to_id']} ({et.value})"}
        cautions = _edge_cautions(graph, op["from_id"], op["to_id"])
        if cautions:
            result["caution"] = cautions
        return result

    if verb == "delete_edge":
        et, err = _parse_edge_type(op["edge_type"])
        if err:
            return {"index": index, "op": verb, "ok": False, "error": err["error"]}
        if et == EdgeType.SCOPE:
            return {"index": index, "op": verb, "ok": False,
                    "error": "SCOPE edges are managed by parent_id; delete the child component instead"}
        edge = Edge(et, op["from_id"], op["to_id"])
        if edge.edge_id not in graph.edges:
            return {"index": index, "op": verb, "ok": False, "error": f"edge '{edge.edge_id}' not found"}
        _remove_edge(graph, graph.edges[edge.edge_id])
        return {"index": index, "op": verb, "ok": True, "id": edge.edge_id}

    if verb == "mark_implemented":
        cid = op["component_id"]
        if cid not in graph.components:
            return {"index": index, "op": verb, "ok": False, "error": f"component '{cid}' not found"}
        sha = _repo_head() if graph is GRAPH else None
        _mark_implemented(graph, cid, sha=sha)
        return {"index": index, "op": verb, "ok": True, "id": cid}

    return {"index": index, "op": verb, "ok": False, "error": f"unknown op '{verb}'"}


@mcp.tool()
def batch_mutate(ops: list[dict]) -> dict:
    """Apply a list of authoring operations atomically: validate all against an
    in-memory scratch copy first, then write to the real graph and persist once.

    Supported op verbs and required fields:
      propose_component  -- component_id, description, processing, input_types,
                            output_types, z_level (+ optional: external, parent_id,
                            locations, feature)
      update_component   -- component_id, fields (dict of changes)
      delete_component   -- component_id
      propose_edge       -- edge_type, from_id, to_id
      delete_edge        -- edge_type, from_id, to_id
      mark_implemented   -- component_id

    Safety guarantees:
      ATOMIC   -- any hard error aborts the whole batch; nothing is written.
      ORDERED  -- ops run in submission order, so parent before child in a
                  propose sequence works; delete children before parents.
      DEFERRED -- mid-batch type gaps or hanging edges from a temporary delete
                  are NOT checked per-op; only the final warning count matters.

    Returns {ok, applied_count, results:[{index, op, ok, id?, error?}],
    active_warnings} on success, or {ok:false, results:[...], applied_count:0}
    with the first failing op on any hard error."""
    if GRAPH is None:
        return _no_active()

    _KNOWN = {
        "propose_component", "update_component", "delete_component",
        "propose_edge", "delete_edge", "mark_implemented",
    }
    _REQUIRED: dict[str, list[str]] = {
        "propose_component": ["component_id", "description", "processing", "input_types", "output_types", "z_level"],
        "update_component":  ["component_id", "fields"],
        "delete_component":  ["component_id"],
        "propose_edge":      ["edge_type", "from_id", "to_id"],
        "delete_edge":       ["edge_type", "from_id", "to_id"],
        "mark_implemented":  ["component_id"],
    }

    # Phase 1 -- pre-flight: schema + delete-orphan guard --------------------
    deleted_ids: set[str] = set()
    for i, op in enumerate(ops):
        verb = op.get("op")
        if not verb:
            return {"ok": False, "applied_count": 0, "results": [
                {"index": i, "op": None, "ok": False, "error": "missing 'op' field"}]}
        if verb not in _KNOWN:
            return {"ok": False, "applied_count": 0, "results": [
                {"index": i, "op": verb, "ok": False, "error": f"unknown op '{verb}'"}]}
        for req in _REQUIRED[verb]:
            if req not in op:
                return {"ok": False, "applied_count": 0, "results": [
                    {"index": i, "op": verb, "ok": False, "error": f"missing required field '{req}'"}]}
        if verb == "delete_component":
            cid = op["component_id"]
            c = GRAPH.components.get(cid)
            if c:
                unguarded = [ch for ch in c.children if ch not in deleted_ids]
                if unguarded:
                    return {"ok": False, "applied_count": 0, "results": [
                        {"index": i, "op": verb, "ok": False,
                         "error": f"'{cid}' has children {unguarded} not deleted earlier in this batch"}]}
            deleted_ids.add(cid)

    # Phase 2 -- dry-run on a deep copy --------------------------------------
    scratch = copy.deepcopy(GRAPH)
    results: list[dict] = []
    for i, op in enumerate(ops):
        r = _apply_batch_op(scratch, i, op)
        results.append(r)
        if not r["ok"]:
            return {"ok": False, "applied_count": 0, "results": results}

    # Phase 3 -- commit to live graph + single persist -----------------------
    for i, op in enumerate(ops):
        _apply_batch_op(GRAPH, i, op)
    _persist()
    return {"ok": True, "applied_count": len(ops), "results": results, "active_warnings": _active_count()}


# ===========================================================================
# Translation orchestration tools -- the operator's runtime surface for driving
# a codebase translation. Thin wrappers over translator/lift.py + verify.py that
# operate on the active graph (GRAPH) and a cached skeleton of the target tree.
# ===========================================================================
def _no_skeleton() -> dict:
    return {
        "error": (
            "No translation skeleton is cached. Call "
            "translate_prepare(source_root) first to ingest and skeleton the "
            "target tree before coverage/stitch/verify."
        )
    }


@mcp.tool()
def translate_prepare(source_root: str, scope: Optional[str] = None) -> dict:
    """Start (or step into) a region of a codebase translation. Ingests and
    skeletons the target tree, caches the skeleton for later coverage/stitch/
    verify calls, and returns the region worksheet the operator needs to author
    this region by hand.

    - source_root: path to the code being translated (e.g. "translator").
    - scope: None for the whole-codebase top pass (author the z=0 subsystems);
      a path-prefix (e.g. "translator/") to slice one region for a focused pass.

    Returns a dict with: the region summary (in-region symbol count, boundary
    stubs), the grouping worksheet (call-graph clusters, entrypoints, fan-in
    hubs, pure leaves), contract hints (with needs_naming flags for erased
    dict/primitive contracts), and flow candidates (producer->consumer edges
    plus blind-spot flags). You then AUTHOR via propose_component/propose_edge --
    nothing here writes to the graph. See translator/LIFT_PROTOCOL.md."""
    if GRAPH is None:
        return _no_active()
    global _TRANSLATE_SKELETON, _TRANSLATE_ROOT
    ctx = _lift.prepare_lift(source_root, scope)
    _TRANSLATE_SKELETON = ctx.skeleton
    _TRANSLATE_ROOT = source_root
    region = ctx.region
    return {
        "source_root": source_root,
        "region": {
            "scope": region.scope,
            "in_region_symbols": len(region.symbols),
            "boundary_stubs": [dataclasses.asdict(b) for b in region.boundary_stubs],
            "intra_region_call_edges": len(region.call_edges),
            "intra_region_dataflow_edges": len(region.dataflow_edges),
        },
        "worksheet": dataclasses.asdict(ctx.abstraction),
        "contract_hints": {sid: dataclasses.asdict(h) for sid, h in ctx.contracts.items()},
        "flow_candidates": [dataclasses.asdict(f) for f in ctx.flows],
        "next_step": (
            "Author this region now via propose_component / propose_edge: group "
            "symbols by responsibility (never 1:1 with functions), name erased "
            "contracts, anchor every node to real locations, wire FLOW from the "
            "candidates. Then call translate_coverage() to see what remains."
        ),
    }


@mcp.tool()
def translate_coverage() -> dict:
    """The resume/progress query: which skeleton symbols are not yet covered by
    any authored node, grouped by module. An empty report (is_complete=True)
    means every region is done. Requires a cached skeleton (translate_prepare).

    Returns {by_module, total_uncovered, total_leaf, covered, is_complete}."""
    if GRAPH is None:
        return _no_active()
    if _TRANSLATE_SKELETON is None:
        return _no_skeleton()
    return dataclasses.asdict(_lift.coverage_tracker(GRAPH, _TRANSLATE_SKELETON))


@mcp.tool()
def translate_stitch() -> dict:
    """Once coverage is empty, surface skeleton call/dataflow edges whose endpoint
    symbols were anchored in different authored subtrees -- cross-region edge
    candidates for the operator to confirm as REFERENCE or FLOW. Requires a
    cached skeleton (translate_prepare).

    Returns {"stitch_candidates": [{from_symbol, to_symbol, from_component,
    to_component, from_root, to_root, kind, suggested_edge_type}]}."""
    if GRAPH is None:
        return _no_active()
    if _TRANSLATE_SKELETON is None:
        return _no_skeleton()
    candidates = _lift.stitch_reconciler(GRAPH, _TRANSLATE_SKELETON)
    return {"stitch_candidates": [dataclasses.asdict(s) for s in candidates]}


@mcp.tool()
def translate_verify() -> dict:
    """Audit the authored graph against the cached skeleton: a trust report with
    anchor / coverage / precision / contract / grounding / externals findings and
    an overall trust score. Requires a cached skeleton (translate_prepare).

    Returns {trust_score, trust_breakdown, anchor_findings, coverage_findings,
    precision_findings, contract_findings, grounding_findings, external_findings}.
    The graph and skeleton themselves are omitted from the response (the operator
    already holds the graph)."""
    if GRAPH is None:
        return _no_active()
    if _TRANSLATE_SKELETON is None:
        return _no_skeleton()
    vm = _verifylib.verify(GRAPH, _TRANSLATE_SKELETON)
    return {
        "trust_score": vm.trust_score,
        "trust_breakdown": vm.trust_breakdown,
        "anchor_findings": [dataclasses.asdict(f) for f in vm.anchor_findings],
        "coverage_findings": [dataclasses.asdict(f) for f in vm.coverage_findings],
        "precision_findings": [dataclasses.asdict(f) for f in vm.precision_findings],
        "contract_findings": [dataclasses.asdict(f) for f in vm.contract_findings],
        "grounding_findings": [dataclasses.asdict(f) for f in vm.grounding_findings],
        "external_findings": [dataclasses.asdict(f) for f in vm.external_findings],
    }


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
def armature_translate() -> str:
    """Translation mode: annotate a codebase and commit it as an Armature graph."""
    return (
        "You are TRANSLATING a codebase into an Armature graph. The skeleton is\n"
        "mechanical ground truth; YOU make every semantic judgment and author the\n"
        "graph by hand via propose_component / propose_edge. The translate_* tools\n"
        "only surface evidence and track progress -- they never write to the graph.\n\n"
        "Open or create the target graph first (open_graph / new_graph). Then run\n"
        "the recursive, scope-bounded protocol -- progress lives in the GRAPH, so\n"
        "any step can be resumed in a fresh session:\n\n"
        "  1. TOP PASS -- translate_prepare(root)  (scope=None, whole codebase)\n"
        "     Read the worksheet (clusters, entrypoints, fan-in hubs) and author the\n"
        "     handful of z=0 SUBSYSTEMS plus the FLOW edges between them. Do not\n"
        "     descend yet -- these roots define the subtrees regions will fill.\n\n"
        "  2. REGION PASSES -- translate_prepare(root, scope='subdir/') per region\n"
        "     For each region (a subdir prefix): group symbols BY RESPONSIBILITY --\n"
        "     never 1:1 with functions. Name erased contracts where a hint has\n"
        "     needs_naming=True (dict/primitive I/O). Wire FLOW from the flow\n"
        "     candidates, and hand-wire the blind-spot flags. Anchor EVERY node to\n"
        "     real file locations.\n\n"
        "  3. COVERAGE LOOP -- translate_coverage()\n"
        "     Lists uncovered symbols by module. Loop back to step 2 for the next\n"
        "     region until is_complete=True (empty report).\n\n"
        "  4. STITCH PASS -- translate_stitch()\n"
        "     Surfaces call/dataflow edges whose endpoints landed in different\n"
        "     subtrees. Confirm each as REFERENCE (calls) or FLOW (true siblings)\n"
        "     via propose_edge. This is the only cross-region reconciliation step.\n\n"
        "  5. VERIFY -- translate_verify()\n"
        "     Returns the trust report (anchor/coverage/contract/grounding findings\n"
        "     + score). Resolve findings, then re-run until clean.\n\n"
        "Authoring rules (types as short concept strings, group by responsibility,\n"
        "anchor everything, one component at a time) are in translator/LIFT_PROTOCOL.md\n"
        "-- read it before authoring.\n\n"
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
        "TREAT `alerts` AND `caution`s AS BLOCKING. When get_work_context leads with\n"
        "`alerts`, or propose_edge returns a `caution`, a node you are building against\n"
        "is planned (a stub -- its code does NOT exist), stale, or drifted. Do not\n"
        "assume its declared outputs work at runtime: surface the alert to the human\n"
        "and decide together whether to build it first, mock it, or proceed knowingly.\n\n"
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
