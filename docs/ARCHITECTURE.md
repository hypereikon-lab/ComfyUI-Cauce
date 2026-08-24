# Architecture

CAUCE is a small native ComfyUI operation pack. It does not own creative
project state or the laboratory runtime.

## Dependency direction

```text
ComfyUI graph
  -> cauce_nodes: socket definitions and JSON reports
      -> cauce: mathematical operations and H3 adapters
          -> NumPy, PyTorch, safetensors, official ComfyUI runtime hooks
```

Bindings may translate ComfyUI values into function arguments, but mathematics
must live in `cauce/`. Core modules must not import browser, tunnel, Manager, or
queue concerns.

## Modules

```text
cauce/
  contracts.py         deterministic hashing + seam schema
  timebase.py          H3 frame/token geometry
  h3.py                AV stream validation + mask capability probe
  continuity.py        causal continuation and decoded range slicing
  seams.py             temporal inpaint planning, masks, and splice
  motion.py            coordinate maps, fields, and sampling
  persistence.py       atomic H3 AV latent save/load
  sigma_transport.py   experimental sampler transport

cauce_nodes/
  continuity.py
  seams.py
  motion.py
  persistence.py
  research.py
```

## Stable contracts

### `CAUCE_MAP`

A motion map is a versioned dictionary containing:

```text
grid       [T,H,W,2] inverse source coordinates
validity   [T,H,W]   sampling confidence/support
fps
operation
parameters
tensor_hash
```

Coordinates use PyTorch normalized `align_corners=False` space. Maps are
resized and composed before final sampling whenever possible.

### `CAUCE_VECTOR_FIELD`

A vector field contains time-varying velocities plus explicit fps, duration,
geometry, and parameters. Integration produces a `CAUCE_MAP`.

### `CAUCE_SEAM`

A seam plan contains only the geometry required for one temporal edit:

- source frame counts;
- working frame count;
- cut and repair ranges;
- H3 sampling range;
- incoming/outgoing guide ranges;
- accepted/splice range;
- deterministic content hash.

It is not global project state. Its inputs remain opaque frames.

### H3 AV latent

H3 samples contain two nested tensors:

```text
visual stream             [B,C,Tv,H,W]
structural-audio stream   [B,C,Ta,F]
```

Visual operations preserve or freeze the second stream. Persistence stores both
tensors atomically in safetensors.

## Stable surface

The 19 stable nodes divide into:

```text
3  continuation
4  temporal inpainting
10 motion maps
2  persistence
--
19
```

Official model loaders, FL2VA/Ref2VA conditioning, guide application, samplers,
VAE encode/decode, and video saving remain official ComfyUI nodes.

Native H3 per-row denoise masks are an official ComfyUI runtime capability.
CAUCE owns only the temporal/spatial support geometry and the construction of a
valid nested AV mask; it does not patch the H3 model.

## Research surface

Five nodes are registered under `CAUCE/Research`. Their code paths are kept
callable so experiments remain reproducible, but they are not dependencies of
the stable operations.

Promotion requires:

1. identity control;
2. small active intervention;
3. decode integrity;
4. measured causal effect;
5. repeated success across representative material;
6. documented resource envelope.

## No shipped graph suite

CAUCE currently ships operations and documentation only. Graphs are composed
from official nodes and CAUCE primitives after the required experiment or
production contract is specified. This keeps the package independent of one
prompt, model filename, resolution, or editorial structure.
