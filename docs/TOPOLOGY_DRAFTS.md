# Operation topology dossiers

`operations/topologies/` contains deterministic, non-executable graph designs
for every operation in the semantic catalog. They answer four narrow questions
before a live ComfyUI session is available:

1. which node class performs each stage;
2. whether that stage belongs to official ComfyUI, vanilla ComfyUI, or CAUCE;
3. which ports and explicit parameters connect the stages;
4. which live checks still prevent the design from becoming a reusable graph.

They are not workflow JSON. A topology has symbolic node keys and named ports,
but deliberately lacks ComfyUI node ids, positions, widgets, link ids, and an
API prompt object. Its required state is `offline-draft`.

## Current dossiers

| Operation | Draft variant | Principal composition |
| --- | --- | --- |
| `generate.keyframed` | `first-last` | official `MiniMaxH3ImageToVideo` with explicit endpoint images |
| `generate.from_references` | `video-reference` | official `MiniMaxH3ReferenceToVideo` with image/video references |
| `generate.with_guides` | `multi-anchor` | ordered official `MiniMaxH3AddGuide` conditioning chain |
| `continue.native_av` | `characterized-layout` | official sampling plus CAUCE AV window/span/append primitives |
| `connect.two_sided_guides` | `default` | decoded left/right ranges, two official guides, exact assembly |
| `reference.transform` | `affine` | deterministic CAUCE coordinate map and decoded-image warp |
| `frames.assemble` | `ordered-concatenation` | exact CAUCE decoded ranges plus vanilla image batching |

This catalog is exhaustive: tests require exactly one topology dossier for each
current semantic operation. Additional variants may be added only after the
catalog schema is intentionally extended beyond that one-draft invariant.

## What is validated offline

`cauce.topologies` checks:

- operation id, version, and declared variant;
- complete coverage of the operation catalog;
- node ownership and required graph-contract node classes;
- edge, binding, and output references;
- exact agreement between topology outputs and operation outputs;
- presence of explicit live gates;
- absence of undeclared topology files.

The test suite additionally loads the actual CAUCE node registry. Every edge,
binding, and operation output that touches a CAUCE node must match that node's
current `INPUT_TYPES` and `RETURN_NAMES`. This catches drift in our own code.
Official and vanilla ports are intentionally not asserted from memory; they are
validated against the laboratory runtime's captured `/object_info`.

## Materialization boundary

The progression is:

```text
offline topology dossier
  -> manually composed active UI graph in live ComfyUI
  -> Workspace Control paired UI/API export
  -> Runtime Control schema/hash/round-trip validation
  -> human graph review and minimal execution
  -> variant-scoped CAUCE UI/API artifact pair
```

No topology file may be copied or renamed into an executable artifact. The live
graph remains the authority for numeric node ids, exact upstream schemas, model
selectors, sampler defaults, and UI serialization.

## Live gates

Every plan records operation-specific gates. At minimum, materialization needs:

- a fresh full-runtime manifest containing the same session's `/object_info`;
- exact upstream class and socket validation;
- one active graph with no stale or bypassed experimental branches;
- paired UI and API hashes from that same graph;
- a guarded parameterization that round-trips to the captured API hash;
- queue execution and exact history/output resolution;
- visual assessment wherever the operation makes a behavioral claim.

Passing deterministic tests therefore means “the design is internally
consistent,” not “H3 executed it” or “the result is visually accepted.”
