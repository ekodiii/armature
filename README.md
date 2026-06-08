# Armature

A graph-based project organization system that lets humans and LLMs plan,
navigate, and execute work on large engineering systems. **The graph is the
design artifact.** Instead of dumping a whole codebase into an LLM's context, the
LLM navigates a queryable graph over MCP — fetching only what it needs, one zoom
level at a time. The graph holds all memory between context wipes.

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

### Configuration

The server is configured entirely through environment variables:

| Variable                | Default                        | Purpose |
|-------------------------|--------------------------------|---------|
| `ARMATURE_GRAPH_PATH`   | `./armature_graph.yaml`        | where the graph YAML for this project lives |
| `ARMATURE_PROJECT_ROOT` | directory of the graph file    | root that `get_component_code` resolves `FileLocation.path` against |

Point `ARMATURE_GRAPH_PATH` at the project you're modeling so each project keeps
its own version-controllable graph.

---

## Registering with a client

### Claude Code (CLI)

```bash
claude mcp add armature \
  --env ARMATURE_GRAPH_PATH=/abs/path/to/your-project/armature_graph.yaml \
  --env ARMATURE_PROJECT_ROOT=/abs/path/to/your-project \
  -- uv run --directory /abs/path/to/armature python server.py
```

### Claude Desktop (JSON)

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "armature": {
      "command": "uv",
      "args": ["run", "--directory", "/abs/path/to/armature", "python", "server.py"],
      "env": {
        "ARMATURE_GRAPH_PATH": "/abs/path/to/your-project/armature_graph.yaml",
        "ARMATURE_PROJECT_ROOT": "/abs/path/to/your-project"
      }
    }
  }
}
```

Any MCP client that speaks stdio works the same way: run `uv run python server.py`
with the two env vars set.

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

## Two modes

Same tools, different framing — pick based on whether a graph already exists.

**Authoring** (graph does not exist): call `new_graph()`, then define **all** `z=0`
components and their `FLOW` edges before decomposing anything. Work top-down, one
level at a time. A context wipe between levels is expected — re-fetch the component
you're decomposing to re-orient.

**Editing** (graph exists): fetch the relevant subgraph first, make only changes
within the scope of the task, verify edges after each write.

**Execution protocol (both modes):** ORIENT → PLAN → WRITE → VERIFY → REPEAT.
Fetch before touching, propose one component at a time, re-fetch to confirm edges
and active warnings are clean.

---

## Tool reference

**Graph**
- `new_graph(overwrite=False)` — create a fresh graph; refuses to clobber an
  existing one unless `overwrite=True`.
- `get_graph_stats()` — component/edge counts, max `z_level`, active warning count.

**Read**
- `get_component(component_id)`
- `get_neighbors(component_id, edge_type, upstream=False)`
- `get_references(component_id, incoming=False)` — weak `REFERENCE` links only
- `search_components(query)` — case-insensitive substring match (embedding search TBD)
- `get_subgraph(component_id, depth=-1)` — SCOPE subtree
- `get_path(from_id, to_id)` — shortest `FLOW` path
- `get_impact(component_id)` — upstream/downstream `FLOW` reachability
- `get_active_warnings()`
- `get_component_code(component_id)` — source from the component's `locations`

**Write** (validated; the graph is saved on success)
- `propose_component(component_id, description, processing, input_types, output_types, z_level, external=False, parent_id=None, locations=None)`
- `update_component(component_id, fields)` — structural fields are protected
- `delete_component(component_id)` — refuses if the component still has children
- `propose_edge(edge_type, from_id, to_id)`
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
