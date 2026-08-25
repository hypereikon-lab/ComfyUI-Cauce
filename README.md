# CAUCE

CAUCE is a native ComfyUI standard library for deterministic media ranges,
MiniMax H3 audiovisual-latent structure, reference-media coordinate maps, and
bounded latent persistence.

Nodes expose low-level operations. Workflow graphs assign them production
meaning. CAUCE does not own H3 model loading, prompts, conditioning from decoded
media, sampling, decoding, scheduling, or a second interface.

CAUCE also defines named, typed operations composed from those primitives and
official/vanilla ComfyUI nodes. An operation is a reusable graph contract, not
a monolithic custom node and not a claim that CAUCE implemented every model
capability present in the graph.

## Node surface

The package registers 24 nodes:

- 1 exact decoded-range node under `CAUCE/Assembly`;
- 7 packed H3 AV operations under `CAUCE/H3 AV Latent`;
- 4 target/guide/reference/conditioning nodes under `CAUCE/H3 Planning`;
- 10 coordinate-map and image-warp nodes under `CAUCE/Motion Maps`;
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
continue.native_av
connect.two_sided_guides
reference.transform
frames.assemble
```

Each operation records typed inputs/outputs, graph-stage ownership, constraints,
artifact state, and evidence. Every current operation also has a validated
non-executable topology dossier. None currently ships as a reusable UI/API
graph pair; materialization requires live paired validation first.

## Install

Install this repository as a ComfyUI custom node and restart the ComfyUI Python
process. CAUCE declares no additional Python package.

```text
https://github.com/hypereikon-lab/ComfyUI-Cauce
```

## Documentation

- [Documentation index](docs/INDEX.md)
- [Architecture and boundaries](docs/ARCHITECTURE.md)
- [Native H3 workflow recipes](docs/NATIVE_H3_WORKFLOWS.md)
- [Semantic operations](docs/OPERATIONS.md)
- [Operation topology dossiers](docs/TOPOLOGY_DRAFTS.md)
- [Operation data contracts](docs/DATA_CONTRACTS.md)
- [Node catalog](docs/NODE_CATALOG.md)
- [Motion-reference maps](docs/MOTION_MAPS.md)
- [Validation protocol](docs/VALIDATION.md)
- [Remote ComfyUI runtime](docs/REMOTE_COMFY_RUNTIME.md)

## Local verification

```bash
python3 -m compileall -q cauce cauce_nodes
python3 -m unittest discover -s tests -v
git diff --check
```
