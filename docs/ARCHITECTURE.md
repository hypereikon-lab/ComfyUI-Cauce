# Architecture

CAUCE is a thin native ComfyUI package. It exposes deterministic operations;
ComfyUI graphs provide orchestration and official MiniMax nodes provide model
conditioning and inference.

```text
decoded media ---------------- exact ranges / reference maps ------ CAUCE
packed H3 AV latent ----------- layout / span / guide / append ---- CAUCE
                                      |
                                      v
official H3 conditioning -> official guider/sampler -> official decode
                                      |
                                      v
decoded range selection / native AV persistence ------------------ CAUCE
```

## Modules

```text
cauce/
  assembly.py      exact decoded-frame selection
  av_latent.py     absolute AV layouts, synchronized spans, guides, append
  contracts.py     canonical JSON, schemas, and content hashes
  h3.py            packed audiovisual-latent validation
  motion.py        coordinate maps, fields, image-space sampling
  persistence.py   atomic packed audiovisual-latent save/load
  timebase.py      exact H3 frame/video-token/audio-token arithmetic

cauce_nodes/
  assembly.py      one decoded-range binding
  av_latent.py     six H3 AV bindings
  motion.py        ten reference-map bindings
  persistence.py   two persistence bindings
```

## H3 AV contracts

`CAUCE_H3_AV_LAYOUT` is serializable and content-addressed. It records one
absolute 24 fps window and its exact visual/audio token boundaries.

`CAUCE_H3_AV_SPAN` carries synchronized video and structural-audio tensor
slices plus their absolute frame range. It intentionally is not exposed as a
standalone `LATENT`: a subrange may inherit a nonzero timeline origin and must
not silently reset its 40 Hz audio clock.

The six operations remain orthogonal:

```text
inspect -> plan -> allocate -> extract -> add guide -> append
```

None loads a model, creates a prompt, selects a sampler, samples, decodes, or
claims a visual objective.

## Motion-map contract

Motion maps use inverse pullback coordinates (`target -> source`) normalized for
PyTorch `grid_sample(..., align_corners=False)`. Maps carry validity and hashes
and can be composed before a single decoded-image sample. Whether a resulting
reference controls H3 as intended is a separate empirical question.

## Dependency policy

Core mathematics remains importable without ComfyUI. PyTorch and ComfyUI are
lazy-imported only at tensor/runtime boundaries. The package adds no pip
dependencies and never owns the GPU software stack.
