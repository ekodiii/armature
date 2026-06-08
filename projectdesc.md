# Armature

A graph-based project organization system that lets humans and LLMs plan, navigate, and execute work on large engineering systems. The graph is the design artifact. LLMs navigate it via MCP tools instead of holding entire codebases in context.

---

## Core Concept

Every engineering system decomposes into components with typed inputs, typed outputs, and a processing description. Armature makes this structure explicit and queryable. Instead of dumping files into an LLM's context, the LLM navigates a graph -- fetching only what it needs, when it needs it.

The graph is fractal. Any component can be decomposed into subcomponents. Any subcomponent is itself a valid component. The tree is infinitely decomposable and the LLM controls its own zoom level.

---

## The Graph

**Three dimensions:**
- XY plane -- FLOW edges, data moving between components at the same level
- Z axis -- SCOPE edges, parent contains child, abstraction depth
- No diagonal *contract* edges. Subcomponents connect to siblings at their own level. Parents expose ports upward through their own input/output types.

**Three edge types:**
```python
class EdgeType(Enum):
    FLOW      = "FLOW"       # data movement between siblings, same z-level
    SCOPE     = "SCOPE"      # parent to direct child only, never skip levels
    REFERENCE = "REFERENCE"  # weak annotation link, may cross parents/z-levels
```

**REFERENCE edges (weak links).** FLOW is strictly sibling-only, so once `A -> B`
is decomposed there is no FLOW edge that can express that `A`'s exit child feeds
`B`'s entry child -- they have different parents. REFERENCE is the deliberate
escape valve: a directed, second-class edge that records cross-boundary data flow
for traceability. It is the *only* edge allowed to cross parents and z-levels, and
it is invisible to every contract -- coverage, cycle detection, and the FLOW walks
behind `get_path`/`get_impact` all ignore it. Surfaced only via `get_references`.
Never required; pure annotation.

**Component structure:**
- one input port (list of declared types)
- one output port (list of declared types)
- every type must map to a downstream consumer or fire a warning
- single type per port is the ideal, lists are allowed, subcomponents handle complexity
- external: bool -- boundary nodes (ffmpeg, database drivers, UI input, third party APIs). never decomposed. treated as atomic.
- components track their own edges_in, edges_out, children, parent_id directly
- locations: list[FileLocation] -- optional file mapping. not required for high level components, effectively required for leaf nodes.

**FileLocation:**
```python
@dataclass
class FileLocation:
    path:       str           # relative to project root
    start_line: Optional[int] = None
    end_line:   Optional[int] = None
```
a list because a component can span multiple files. unlocks: direct code reading via mcp, precise edits without loading whole files, gui jump-to-file, and eventually type validation against real function signatures.

**Entry and exit nodes:**
In a subprocess (multi-node child pipeline), validation checks only:
- entry nodes (children with no incoming FLOW from siblings) must cover parent input types
- exit nodes (children with no outgoing FLOW to siblings) must cover parent output types
- internal nodes are not validated against the parent

**Type system:**
Types are strings. No formal registry. Convention over enforcement -- same as variable naming in code. Context and scope keep collisions unlikely. The tool surfaces contracts, it does not enforce a type calculus.

**Storage:**
Hanging outputs (output types with no downstream consumer) fire warnings. Storage components are just regular external components. When a hanging output gets wired to a storage component the warning disappears.

**Emergent behavior:**
Modeled as external leaf nodes. Types declared at the boundary, internals opaque. Same treatment as any black box.

**Cross-cutting concerns:**
Auth is a FLOW node with AuthSuccess and AuthFailure output types. Logging is a hanging output that maps to a storage component. Neither requires special treatment.

---

## File Structure

```
armature/
├── models.py       -- dataclasses: Component, Edge, Graph, Warning, FileLocation, EdgeType
├── store.py        -- raw graph operations, no validation
├── validator.py    -- validation functions, return error strings
├── graph_warnings.py -- graph-wide warning system (named to avoid shadowing stdlib `warnings`)
├── operations.py   -- read-only traversal: get_neighbors, get_references, get_subgraph, get_path, get_impact
├── serializer.py   -- yaml in/out
├── writer.py       -- validated writes: propose_component, update_component, delete_component, propose_edge
└── server.py       -- MCP server (FastMCP): graph/read/write tools, persistence, mode instructions
```

Each project's graph lives in the project directory as a yaml file. Version controllable, human readable.

---

## Key Design Decisions

**Ports are folded into components.** input_types and output_types are lists on the component itself. No separate Port class -- ports as independent objects were only needed for diagonal edges, which we eliminated.

**Edges are a class with a derived id:**
```python
@dataclass
class Edge:
    edge_type: EdgeType
    from_id:   str
    to_id:     str

    @property
    def id(self) -> str:
        return f"{self.from_id}__{self.to_id}__{self.edge_type.value}"
```

**Components track their own connections.** edges_in, edges_out, and children are lists of ids on the component. Redundant with the graph's edge dict but makes traversal O(1) from the component itself. These are *derived state*: rebuilt in memory on load from the edge list + parent_id, never serialized. validate_consistency() checks they stay in sync.

**Serialization stores only authoritative state.** parent_id is the source of truth for the hierarchy, so SCOPE edges are regenerated from it on load rather than written. edges_in/edges_out/children are rebuilt on load. Warnings are recomputed on load; only the *ignore decisions* (id + reason) persist. This keeps the yaml roughly an order of magnitude smaller than serializing every derived field.

**Implementation status is derived, not a manual tag.** Each component stores implemented_version (the version its code was last written against). status falls out: planned (None), implemented (== version), stale (< version). Because update_component bumps version, editing a spec auto-flips its code to stale -- the "changed" signal can never drift out of sync. mark_implemented() advances the marker; it is the bridge between graph (spec) and code (implementation).

**Store is pure operations, no validation.** writer.py gates all writes through validator.py. store.py just does the mechanical work.

**Warnings are non-blocking; only ignore decisions persist.** Every warning includes enough context to decide whether to ignore it. Warnings themselves are derived (recomputed each run); the persisted ignore decisions are re-applied by id so the LLM doesn't re-surface dismissed warnings across sessions.

**Warning types currently implemented:**
- hanging_output -- output type with no downstream consumer
- undefined_types -- empty input or output types on a non-external component
- orphaned_component -- component with no edges at all

**File mapping:**
locations field on Component is optional and empty by default. high level components don't need it. leaf nodes effectively require it for any code-level work. the codebase parser fills this automatically in v2.

---

## MCP Server

The LLM never gets the whole graph. It navigates via tool calls.

**Workspace / graph tools:**
- list_graphs() -- every named graph, path, component count, which is active.
- new_graph(name, path, overwrite) -- create a named graph (central by default, or at a repo path). entry point for authoring mode.
- open_graph(name) -- make an existing graph active. entry point for editing/implementing.
- get_graph_stats() -- name, paths, component/edge counts, max z_level, by_status counts, active warnings count. orientation tool.

**Read tools:**
- get_component(id)
- get_work_context(id) -- the 1-hop frame for safely writing or implementing one node (own contract, receives, must_produce, parent contract, children, references, current code, its warnings). call before every write.
- get_neighbors(id, edge_type, upstream)
- get_references(id, incoming) -- weak REFERENCE links only, separate from FLOW/SCOPE traversal
- search_components(query) -- semantic search over descriptions
- get_subgraph(id, depth)
- get_path(from_id, to_id)
- get_impact(id)
- get_active_warnings() -- compact {id, type, affected} + a per-type legend
- get_component_code(id) -- reads actual file at component's location, returns source. requires locations to be set.
- get_pending_implementation() -- components that are planned or stale (the implementation queue), stale/leaves first.

**Write tools (all validated before commit):**
- propose_component(component)
- update_component(id, fields)
- delete_component(id)
- propose_edge(edge)
- mark_implemented(id) -- record that the code matches the current spec version.
- ignore_warning(warning_id, reason)

**Three modes, same tools, different system prompts (each also an MCP prompt / slash command):**

AUTHORING MODE -- graph does not exist yet:
"Call new_graph() first. Define all z=0 components and their edges before decomposing anything. Do not go deeper until the top level is complete and validated. Work top-down one level at a time. Context wipe between levels is expected -- fetch the component you are decomposing first to re-orient."

EDITING MODE -- graph already exists:
"open_graph(), fetch the relevant subgraph before touching anything. Make only changes within the scope of the current task. Orient, plan, write, verify. Editing a spec bumps version and flips status to stale."

IMPLEMENTING MODE -- turn the graph into code:
"open_graph(), get_pending_implementation() for planned/stale components. For each: get_work_context(id), write the code to honor the contract with your own editor tools, set locations, then mark_implemented(id)."

**LLM execution protocol (all modes):**
1. ORIENT -- call get_work_context(id) for the exact node before touching it; re-pull per change, never write from session memory.
2. PLAN -- reason about what needs to change
3. WRITE -- propose / implement one component at a time
4. VERIFY -- re-fetch what was written, confirm edges and warnings are clean
5. REPEAT until done

---

## Warning System

```python
@dataclass
class Warning:
    id:            str           # deterministic: "{warning_type}__{component_id}"
    warning_type:  str
    message:       str           # why it fired AND why it might be safe to ignore
    affected:      list[str]     # component ids or type names
    ignored:       bool = False
    ignore_reason: Optional[str] = None
```

Warnings live on the graph but are derived: recomputed on every run from the current graph state. Only the ignore decisions (warning id + reason) are serialized, and re-applied by id on load, so ignored state persists across sessions without storing the warnings themselves.

---

## Planning Workflow

**Authoring (graph does not exist):**
Human describes the system in plain language → LLM calls new_graph() → creates all z=0 components and edges first, rough types allowed → context wipe → LLM fetches one z=0 component → decomposes into z=1 subcomponents → context wipe → repeat downward until leaf nodes have atomic types and all edges resolve. Warnings surface gaps at each level.

**Editing (graph exists):**
Human describes a change or task in plain language → LLM fetches relevant subgraph → orients against current state → proposes changes within scope → verifies edges after each write → surfaces warnings for human review.

The graph holds all memory between context wipes. The LLM can always reconstruct where it was by fetching the relevant component. Context never rots because the LLM never holds more than one zoom level at a time.

---

## Roadmap

**v1 (current):**
- single user, single graph per project
- LLM authoring mode (new_graph → top-down decomposition) and editing mode (task scoped changes)
- human reviews and approves at each zoom level
- yaml storage, in-memory graph
- full validation and warning system
- MCP server with read, write, and graph tools including get_component_code

**v2:**
- codebase parser -- static analysis (ast / tree-sitter) for structure + LLM for type summarization
  - pass 1: static analysis downward (structure, edges, z_levels, external detection, file locations)
  - pass 2: LLM summarization upward (type inference, descriptions, bottom to top)
- multi-graph support -- separate graphs per service/layer with an integration graph at the boundary
- graph database backend (neo4j) for scale, yaml stays as human-facing export format
- optimistic concurrency (version field on components already in place)
- cross-domain translation (general-engineering only -- not a concern for single-domain software graphs)
  - problem: a value that is a single typed I/O in one domain (one electronics signal) gets unpacked into many orthogonal concerns in another (payload / clock / parity / flow-control in firmware). representation is not conserved across the seam, so string-equality type matching makes every domain boundary look broken -- both hanging_output and starved_input fire on a perfectly correct edge.
  - not a ports problem: input/output are already lists, so cardinality is fine. the pressure is on two assumptions: (a) one shared type vocabulary across the whole graph, and (b) type continuity across edges.
  - transducer nodes: model a domain boundary as an explicit node (same move as auth/logging), whose declared job is translation. input vocabulary != output vocabulary by design, and continuity warnings are suppressed there only. do NOT relax the one-in/one-out discipline globally.
  - domain tag on components: makes seams detectable (warn on a cross-domain edge with no transducer), scopes type names so the same name in two domains stops colliding, and seeds the multi-graph split.
  - dimensional explosion = decomposition, not a type calculus: bridge the seam with a single composite type and let it unpack as ordinary z-level decomposition inside the target domain. keep types as bare strings; never build structured types.
  - the boundary is also a cross-parent crossing: domains are separate subtrees, so the same node is a leaf (exit) in one domain and a source (root) in another. this is the strongest case for promoting REFERENCE to a first-class interface edge, or for multi-graph with a shared integration boundary -- two domain views agreeing on an interface contract.

**v3:**
- multi-user with task scope as lock primitive
- event sourcing underneath graph mutations
- GUI -- react flow based, z-levels as actual visual depth, warnings panel, click component to jump to file

---

## Open Questions

- embedding search backend for search_components (currently stubbed as substring match)
- how to handle runtime dynamic behavior (feature flags, dynamic routing) in a static graph
- parser pass 3: top-down description refinement after bottom-up summarization -- v2 or post-launch task
- test project: build something real or find an open source project and manually author its graph

---

## Arguments Against Core Concept (resolved)

- **emergent behavior** -- external leaf node, types declared at boundary, internals opaque
- **cross-cutting concerns** -- dissolve when modeled as explicit flow. auth is a node. logging is a side output.
- **type registry** -- convention problem not a tool problem. same as variable naming.
- **atomicity subjectivity** -- acceptable within a team context. consistent mental model matters more than objective definition.
- **llm graph reasoning** -- empirical, only answerable by building and testing.
- **cross-parent flow at depth** -- FLOW is strictly sibling-only, so data crossing a component boundary below z=0 (A's exit child feeding B's entry child) has no FLOW representation. Primary answer: flow maps up to the boundary edge and is re-derived top-down on the way back into the next step. REFERENCE edges add an optional weak link to preserve the concrete cross-boundary trace without weakening the no-diagonal-FLOW invariant.
