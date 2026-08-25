# Architecture

CAUCE is a thin native ComfyUI package. It supplies deterministic data
operations; the official ComfyUI graph remains the orchestration layer.

```text
source media
  |-- exact range/guide selection ----------- CAUCE
  |-- coordinate maps -> warped references -- CAUCE
  v
official H3 conditioning -> sampler -> decode ComfyUI/MiniMax
  v
range acceptance / exact assembly ----------- CAUCE
  v
normal ComfyUI save nodes
```

## Modules

```text
cauce/
  assembly.py      exact decoded-frame selection
  two_sided_window.py  AddGuide window plan, extraction, assembly
  contracts.py     canonical JSON and content hashes
  h3.py            packed audiovisual-latent validation
  motion.py        coordinate maps, fields, image-space sampling
  persistence.py   atomic packed audiovisual-latent save/load
  timebase.py      exact H3 frame/token arithmetic

cauce_nodes/
  assembly.py      one ComfyUI binding
  two_sided_window.py  two ComfyUI bindings
  motion.py        ten ComfyUI bindings
  persistence.py   two ComfyUI bindings
```

## Contracts

The two-sided window plan is serializable and content-addressed. It records source
ranges, guide placement, accepted generated range, and ownership. It contains
no prompt semantics and no hidden process state.

Motion maps use inverse pullback coordinates (`target -> source`) normalized for
PyTorch `grid_sample(..., align_corners=False)`. Maps carry validity and hashes
and can be composed before a single image sample.

Packed H3 persistence saves the visual and structural-audio tensors atomically
as `safetensors`. It does not reinterpret either stream.

## Dependency policy

Core mathematics remains importable without ComfyUI. PyTorch and ComfyUI are
lazy-imported only at tensor/runtime boundaries. The package adds no pip
dependencies and never owns the GPU software stack.
