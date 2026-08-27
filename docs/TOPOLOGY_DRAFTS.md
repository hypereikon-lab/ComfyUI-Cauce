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

`operations/archetypes/catalog.json` adds a narrower structural identity above
the dossiers. It hashes node keys, owners, class types, and exact edges while
excluding bindings and descriptive metadata. The 28 dossiers currently form 25
archetypes. A shared archetype means one eventual paired graph can support
multiple guarded binding profiles; it does not merge their semantic variants.

## Current dossiers

| Operation | Draft variant | Principal composition |
| --- | --- | --- |
| `generate.keyframed` | `text-only`, `first-frame`, `last-frame`, `first-last` | complete official `MiniMaxH3ImageToVideo` endpoint-input matrix |
| `generate.from_references` | `image-reference-match`, `image-reference-max`, `video-reference`, `video-reference-with-guide` | official Ref2VA with optional temporal guide on the same conditioning edge |
| `generate.with_guides` | `single-anchor`, `multi-anchor`, `guide-clip`, `first-last-interior` | official endpoint frames plus exact temporal image/clip guides |
| `continue.native_av` | `keyframe-overlap`, `masked-overlap`, `masked-overlap-future-guide` | explicit native overlap transport around ordinary official sampling |
| `complete.native_av` | `backward-prefix`, `two-sided-infill`, `local-replacement`, `two-source-connection` | native span placement, independent AV masks, optional exact interval replacement |
| `rollback.native_av` | `branch-suffix` | exact native split and optional branch persistence |
| `frames.assemble` | `ordered-concatenation` | exact CAUCE decoded ranges plus vanilla image batching |

This catalog is exhaustive over files and complete over operations: tests
require at least one dossier for every current semantic operation, permit
multiple explicitly named variants, reject duplicate operation/variant pairs,
and reject uncatalogued plan files.

## What is validated offline

`cauce.topologies` checks:

- operation id, version, and declared variant;
- complete coverage of the operation catalog;
- node ownership and required graph-contract node classes;
- edge, binding, and output references;
- exact agreement between topology outputs and operation outputs;
- presence of explicit live gates;
- absence of undeclared topology files.

It also requires every topology to belong to exactly one graph archetype,
rejects mixed structures inside an archetype, and rejects duplicate archetypes
for the same structural signature.

The test suite additionally loads the actual CAUCE node registry. Every edge,
binding, and operation output that touches a CAUCE node must match that node's
current `INPUT_TYPES` and `RETURN_NAMES`. This catches drift in our own code.
Official and vanilla ports are intentionally not asserted from memory; they are
validated against the laboratory runtime's captured `/object_info`.

## Materialization boundary

The progression is:

```text
operation contract
  -> graph archetype
  -> variant binding profile
  -> offline topology dossier
  -> manually composed active UI graph in live ComfyUI
  -> Workspace Control paired UI/API export
  -> Runtime Control schema/hash/round-trip validation
  -> human graph review and minimal execution
  -> variant-scoped CAUCE UI/API artifact pair
```

No topology or archetype file may be copied or renamed into an executable
artifact. The live
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
