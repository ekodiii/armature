# Lift Protocol — operator's recursive guide

The lift stage turns a `CodeSkeleton` (mechanical, location-pinned ground truth)
into a `SemanticModel` (a responsibility-driven Armature graph). The semantic
judgment is yours, the **operator**. `translator/lift.py` is a deterministic
assist layer: it computes evidence, slices work into context-sized regions,
tracks progress in the graph, and checks your output mechanically. It never
calls an LLM, writes descriptions, or decides groupings — you do.

Because the skeleton worksheet stays small even for huge codebases, lift runs as
**recursive scope-bounded descent**. Progress lives in the graph, not your
context, so any session can cut off and a fresh one resumes. There is **no graph
merging** — every session writes into the one open graph.

## 0. Top pass (worksheet only)
Open the target graph. From the compact whole-codebase worksheet
(`prepare_lift(root)` with `scope=None`, or just the module list), author the
**z=0 subsystems** — the handful of top-level responsibilities — and the `FLOW`
edges between them. Do not descend yet. These roots define the subtrees that
regions will fill in.

## 1. Region passes (one fresh session each)
Pick a region (a subdirectory prefix like `"translator/"`, or an explicit symbol
set). In a fresh operator session:

```python
ctx = prepare_lift(root, scope="translator/")   # ingest + skeleton + 4 prep passes
```

`ctx` bundles `region`, `abstraction`, `contracts`, `flows`. Then:

1. **Group by responsibility** — read `ctx.abstraction` (per-module symbol
   lists, call-graph clusters, entrypoints, fan-in hubs, pure leaves). Author
   components that group symbols by *concept*. **Never 1:1 with functions**: a
   node may span many functions, or split one function across concerns. Clusters
   and fan-in hubs are hints, not a blueprint.
2. **Name erased contracts** — for every symbol, `ctx.contracts[id]` is a
   `ContractHint`. Where `needs_naming=True` (dict/primitive-shaped I/O), invent
   a semantic type name for the component's `input_types`/`output_types` instead
   of leaving `dict`/`str`/`list`.
3. **Wire FLOW** — from `ctx.flows`: `dataflow` candidates are strong
   producer→consumer signals; `call` candidates are weaker (often control, not
   data). Entries with `blind_spot=True` mark hidden flows (in-place container
   mutation) the skeleton cannot see — wire those `FLOW` edges by hand.
4. **Anchor everything** — every authored node gets `locations` pointing at the
   real `path`/`start_line`/`end_line` of the symbols it covers. Use
   `intent_describer(region, symbol_id)` for a snippet + neighbours when writing
   a node's what/why.

Author with `propose_component` / `propose_edge`. FLOW only connects siblings
under the same parent; cross-region links are deferred to the stitch pass.

## 2. Coverage loop (the resume mechanism)
After a region lands, run:

```python
coverage_tracker(graph, skeleton)   # -> CoverageReport, uncovered grouped by module
```

A non-empty report tells the next session which modules still need a region.
Loop back to step 1 until `is_complete` (empty). This is how lift survives
context wipes: the graph is the memory.

## 3. Stitch pass (edge reconciliation, not merging)
Once coverage is empty:

```python
stitch_reconciler(graph, skeleton)  # -> StitchCandidates across subtree boundaries
```

Each candidate is a skeleton call/dataflow edge whose endpoints were anchored in
**different** region subtrees. Confirm or reject each as `REFERENCE` (calls) or
`FLOW` (dataflow between true siblings). This is the only "combining" step.

## 4. Consolidate + verify
```python
consolidator(graph, skeleton)       # -> ResidueFlags (merge-up candidates)
verify(graph, skeleton)             # -> VerifiedModel + trust score
```

`consolidator` flags call-tree residue — nodes that merely mirror one symbol
with no contract gain. Merge each into its parent, or justify keeping it.
Resolve residue flags, then run `verify` and close out the remaining findings.
