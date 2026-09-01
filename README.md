# CAUCE

CAUCE is a native ComfyUI custom-node library for deterministic media ranges,
MiniMax H3 audiovisual-latent structure, and bounded latent persistence.

Nodes expose low-level operations. Workflow graphs assign them production
meaning. CAUCE does not own H3 model loading, prompts, conditioning from decoded
media, sampling, decoding, scheduling, or a second interface.

CAUCE also defines named, typed operations composed from those primitives and
official/vanilla ComfyUI nodes. An operation is a reusable graph contract, not
a monolithic custom node and not a claim that CAUCE implemented every model
capability present in the graph.

The thirteen operations form three non-sequential families:

```text
H3 conditioning grammar     keyframed / references / guides / structural control
native H3 AV state algebra  continue / complete / densify / edit / outpaint / refine / regenerate / rollback
decoded media algebra       exact frame assembly
```

Primitives, operations, graph archetypes, binding profiles, workflow pairs,
invocations, runs, and evidence are distinct lifecycle states. The 37 current
operation variants include 35 topology dossiers resolving to 32 structurally
distinct archetypes; variants
that differ only by guarded literals share one graph. See the
[operation model](docs/OPERATION_MODEL.md).

## Node surface

The package registers 28 nodes:

- 2 exact decoded-frame nodes under `CAUCE/Assembly`;
- 17 packed H3 AV operations under `CAUCE/H3 AV Latent`;
- 6 target/guide/reference/control/conditioning nodes under `CAUCE/H3 Planning`;
- 1 exact H3 guide-retime planning node under `CAUCE/Temporal Planning`;
- 2 packed H3 AV save/load nodes under `CAUCE/Persistence`.

## Native AV continuation as graph composition

CAUCE does not provide a monolithic “continue video” node. A continuation graph
is composed explicitly:

```text
previous H3 AV latent
  -> Plan H3 AV Window
  -> Allocate H3 AV Window

previous H3 AV latent
  -> Extract H3 AV Span (tail)
  -> Add H3 AV Span Guide to official positive conditioning
  -> official guider / sampler
  -> Extract H3 AV Span (new suffix)
  -> Append H3 AV Span
```

The layout keeps video at H3's `17k+5` frame grid and structural audio at its
40 Hz token grid against absolute 24 fps frame boundaries. The sampler remains
an ordinary official ComfyUI sampler.

## Operation surface

The current semantic catalog is open-ended and contains:

```text
generate.keyframed
generate.from_references
generate.with_guides
generate.with_control
continue.native_av
complete.native_av
densify.temporal
edit.masked_video
reframe.outpaint_video
refine.video
regenerate.spatial
rollback.native_av
frames.assemble
```

Each operation records typed inputs/outputs, graph-stage ownership, constraints,
artifact state, and evidence. Thirty-five validated, non-executable topology
dossiers cover every current operation, including official keyframe/reference/
guide/control combinations and native AV continuation, completion, temporal densification,
spatial regeneration, replacement, and rollback variants. A content-addressed archetype catalog groups only dossiers
whose nodes and edges are identical. None currently ships as a reusable UI/API graph pair;
materialization requires live paired validation first.

## Install

Install this repository as a ComfyUI custom node and restart the ComfyUI Python
process. CAUCE declares no additional Python package.

```text
https://github.com/hypereikon-lab/ComfyUI-Cauce
```

## Documentation

- [Documentation index](docs/INDEX.md)
- [Architecture and boundaries](docs/ARCHITECTURE.md)
- [Operation families and lifecycle](docs/OPERATION_MODEL.md)
- [Native H3 workflow recipes](docs/NATIVE_H3_WORKFLOWS.md)
- [Current H3 extensions and experiment boundary](docs/H3_EXTENSION_MAP.md)
- [Official H3 structural control](docs/STRUCTURAL_CONTROL.md)
- [Temporal and spatial video enhancement](docs/TEMPORAL_SPATIAL_ENHANCEMENT.md)
- [Semantic operations](docs/OPERATIONS.md)
- [Operation topology dossiers](docs/TOPOLOGY_DRAFTS.md)
- [Operation data contracts](docs/DATA_CONTRACTS.md)
- [Node catalog](docs/NODE_CATALOG.md)
- [Validation protocol](docs/VALIDATION.md)

## Local verification

```bash
python3 -m compileall -q cauce cauce_nodes
python3 -m unittest discover -s tests -v
python3 tools/verify_full.py  # release gate: requires a pre-existing NumPy runtime
git diff --check
```
