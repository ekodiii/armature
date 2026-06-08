# Armature

A graph-based organization system that lets humans and LLMs plan, navigate, and
execute work on large engineering systems of any kind — software, mechanical,
electrical, process, organizational. **The graph is the design artifact.** Instead
of holding an entire system in context, the LLM navigates a queryable graph over
MCP — fetching only what it needs, one zoom level at a time. The graph holds all
memory between context wipes.

For the full design rationale, see [`projectdesc.md`](projectdesc.md).

---

## The model in one screen

Every system decomposes into **components**: a `processing` description, one input
port and one output port (each a list of string **types**), a `z_level`
(abstraction depth — top level is `0`), and an `external` flag for atomic boundary
nodes (DB drivers, third-party APIs, UI input).

Components decompose fractally: a component at `z=N` is decomposed into children at
`z=N+1`, and any child is itself a valid component.

**Three edge types:**

| Edge        | Connects                              | Rule |
|-------------|---------------------------------------|------|
| `FLOW`      | siblings, same `z_level`              | both ends share the same parent — no diagonals |
| `SCOPE`     | parent → direct child                 | exactly one `z_level` down; **auto-created** when you propose a component with a `parent_id` |
| `REFERENCE` | anything → anything                   | the only edge allowed to cross parents/`z_levels`; a weak annotation, invisible to all contracts |

**Validation is layered.** Structural problems (bad `z_level`, non-sibling `FLOW`,
missing parent, duplicate id/edge) **block** a write. Everything softer surfaces as
a non-blocking **warning** you can read and, if intentional, ignore — ignored state
persists across sessions.

Warning types: `hanging_output`, `starved_input`, `undefined_types`,
`orphaned_component`, `coverage_gap`, `flow_cycle`.

---

## Setup

Requires Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Run the MCP server (stdio transport):

```bash
uv run python server.py
```

### Workspace

You keep many **named graphs**, one per system you're modeling. They live in a
central library and are selected by name at runtime — there is **no dependency on a
working directory**, so the exact same setup works on a terminal, a desktop app, or
the web. The model drives it with three tools: `list_graphs()`, `new_graph(name)`,
`open_graph(name)`. One graph is *active* per session, and the last-active one is
remembered across restarts.

```
~/.armature/
├── registry.json          # name -> file path, plus the active graph
└── graphs/
    ├── hvac-redesign.yaml
    └── payment-service.yaml
```

A graph's file can also live **inside a code repo** for version control — pass a
path when creating it (`new_graph("payments", path="./armature_graph.yaml")`) and
it's still opened later by name. Relative `FileLocation` paths (used by
`get_component_code`) then resolve against that repo automatically.

Optional environment variables:

| Variable                | Default          | Purpose |
|-------------------------|------------------|---------|
| `ARMATURE_HOME`         | `~/.armature`    | where the graph library and registry live |
| `ARMATURE_PROJECT_ROOT` | dir of the active graph file | override the root that `get_component_code` resolves paths against |

---

## Registering with a client

The same command works everywhere — call the venv's Python directly, no env vars
and no working-directory assumptions:

### Claude Code (CLI)

```bash
claude mcp add armature -- /abs/path/to/armature/.venv/bin/python /abs/path/to/armature/server.py
```

### Claude Desktop (and any stdio MCP client)

```json
{
  "mcpServers": {
    "armature": {
      "command": "/abs/path/to/armature/.venv/bin/python",
      "args": ["/abs/path/to/armature/server.py"]
    }
  }
}
```

In both cases the model starts by calling `list_graphs()` and then `open_graph` or
`new_graph`. `get_graph_stats` reports the active graph's name and resolved paths
so you can confirm what you're working on at any time.

---

## Making the LLM actually use it for planning

Connecting the server makes the tools **discoverable** — the client injects every
tool's name and description, plus the server's instructions, into the model's
context. That's enough for the model to reach for Armature when a request clearly
matches.

It is **not** enough to make graph-driven planning the *default*. To get that, add
one line to the consuming project's `CLAUDE.md` (or the client's custom
instructions):

> For any architecture, planning, or decomposition task in this repo, drive it
> through the Armature MCP tools rather than free-forming a plan in prose.

Tool descriptions can't set a default policy; only the host's system prompt can.
(A future addition is exposing the two mode prompts below as MCP `prompts`, which
surface as slash commands.)

---

## Three modes

Same tools, different framing. Each is exposed as an MCP **prompt**
(`armature_plan`, `armature_implement`) that loads the right discipline as a slash
command.

**Planning** (`/armature_plan`): all graph work — new systems, changes to existing
ones, decomposing a component deeper. The prompt reads the request and takes the
right path automatically:
- *New system*: `new_graph(name)`, define all `z=0` components and `FLOW` edges
  before decomposing anything, work top-down one level at a time.
- *Change to existing*: `search_components` to find what the request refers to,
  `get_impact` before touching anything, change only what is in scope.
- *Decomposing a component*: `get_work_context(id)` locks you to that component's
  contract — the rest of the graph is invisible until you're done.

**Implementing** (`/armature_implement`): realizing what the graph specifies — code,
hardware, configuration, process, whatever the domain requires. Human-directed: the
LLM responds to what the human asks or reports, it does not autonomously drain the
queue. For each request it finds the right component, establishes scope with
`get_impact`, verifies the artifact against the contract, and calls
`mark_implemented(id)`. If reality doesn't match the spec it stops, surfaces the
discrepancy, and asks whether the artifact or the graph needs to change — mismatches
are never silently reconciled.

**Execution protocol (all modes):** ORIENT → PLAN → WRITE → VERIFY → REPEAT. The
key rule: call **`get_work_context(id)` immediately before every write or
implementation** of a component — it returns just that node's 1-hop frame (its
contract, what it receives/must produce, parent contract, children, current code,
its warnings), so you re-pull scoped context per change instead of holding the whole
graph. Never write from session memory; the graph moves under you.

### Component status

Derived from `version` vs `implemented_version`, so it can't drift:

| status        | meaning |
|---------------|---------|
| `planned`     | no code written yet |
| `implemented` | code matches the current spec version |
| `stale`       | spec was edited after the code was written — needs updating |

`mark_implemented(id)` advances a component to `implemented`; any later edit
auto-flips it back to `stale`.

---

## Tool reference

**Workspace / graph**
- `list_graphs()` — all named graphs, paths, component counts, which is active.
- `new_graph(name, path=None, overwrite=False)` — create a named graph (central by
  default, or at `path` to store it in a repo); refuses to clobber unless `overwrite=True`.
- `open_graph(name)` — make an existing graph active (returns its stats).
- `get_graph_stats()` — active graph's name, paths, component/edge counts, max
  `z_level`, active warning count.

**Read**
- `get_component(component_id)`
- `get_work_context(component_id)` — the 1-hop frame for safely writing/implementing one node; call before every write
- `get_neighbors(component_id, edge_type, upstream=False)`
- `get_references(component_id, incoming=False)` — weak `REFERENCE` links only
- `search_components(query)` — case-insensitive substring match (embedding search TBD)
- `get_subgraph(component_id, depth=-1)` — SCOPE subtree
- `get_path(from_id, to_id)` — shortest `FLOW` path
- `get_impact(component_id)` — upstream/downstream `FLOW` reachability
- `get_active_warnings()`
- `get_component_code(component_id)` — source from the component's `locations`
- `get_pending_implementation()` — components that are `planned` or `stale` (the implementation queue)

**Write** (validated; the graph is saved on success)
- `propose_component(component_id, description, processing, input_types, output_types, z_level, external=False, parent_id=None, locations=None)`
- `update_component(component_id, fields)` — structural fields are protected
- `delete_component(component_id)` — refuses if the component still has children
- `propose_edge(edge_type, from_id, to_id)`
- `mark_implemented(component_id)` — record that the code matches the current spec version
- `ignore_warning(warning_id, reason)`

---

## File structure

```
armature/
├── models.py          -- dataclasses: Component, Edge, Graph, Warning, FileLocation, EdgeType
├── store.py           -- raw graph operations, no validation
├── validator.py       -- validation functions, return error strings
├── graph_warnings.py  -- graph-wide warning system (named to avoid shadowing stdlib `warnings`)
├── operations.py      -- read-only traversal
├── serializer.py      -- yaml in/out
├── writer.py          -- validated writes
└── server.py          -- MCP server (FastMCP)
```

The graph is a single YAML file per project — version-controllable and human
readable.
