# Architecture

CAUCE is a thin native ComfyUI package. It exposes deterministic operations;
ComfyUI graphs provide orchestration and official MiniMax nodes provide model
conditioning and inference.

The repository also owns semantic operation contracts. They describe typed
graph-level functions and node ownership without turning a complete workflow
into a custom node.

```text
decoded media ---------------- exact ranges ----------------------- CAUCE
H3 decoded inputs ------------- target / guide / reference plans -- CAUCE
H3 conditioning --------------- read-only structural inspection --- CAUCE
packed H3 AV latent ----------- layout / span / place / mask / replace -- CAUCE
                                      |       guide / split / append
                                      |
                                      v
official H3 conditioning -> official guider/sampler -> official decode
                                      |
                                      v
decoded range selection / native AV persistence ------------------ CAUCE
```

```text
operation contract
  -> content-addressed graph archetype
      -> guarded binding profile
          -> paired UI/API artifacts, once live-validated
              -> project invocation and run receipt outside this repository
```

The operation catalog is a capability grammar, not a pipeline. It groups
official H3 conditioning, native AV state transformations, and deterministic
decoded-media transformations while keeping their node ownership explicit.
See [Operation model](OPERATION_MODEL.md).

## Modules

```text
cauce/
  assembly.py      exact decoded-frame selection
  av_latent.py     absolute AV layouts, spans, placement, masks, replacement,
                   native guides, split, and append
  contracts.py     canonical JSON, schemas, and content hashes
  conditioning.py  read-only H3 conditioning metadata inspection
  h3.py            packed audiovisual-latent validation
  h3_inputs.py     target, guide-clip, and reference-clip planning
  persistence.py   atomic packed audiovisual-latent save/load
  timebase.py      exact H3 frame/video-token/audio-token arithmetic

cauce_nodes/
  assembly.py      one decoded-range binding
  av_latent.py     eleven H3 AV bindings
  planning.py      four H3 input/conditioning planning bindings
  persistence.py   two persistence bindings

operations/
  catalog.json     open semantic operation catalog
  archetypes/      structural grouping over topology dossiers
  schema.json      operation data contract
  specs/           typed graph-level operation definitions
  evidence/        structured evidence that is not an importable graph
```

## H3 AV contracts

`CAUCE_H3_AV_LAYOUT` is serializable and content-addressed. It records one
absolute 24 fps window and its exact visual/audio token boundaries.

`CAUCE_H3_AV_SPAN` carries synchronized video and structural-audio tensor
slices plus their absolute frame range. It intentionally is not exposed as a
standalone `LATENT`: a subrange may inherit a nonzero timeline origin and must
not silently reset its 40 Hz audio clock.

The eleven AV-state operations remain orthogonal:

```text
inspect -> plan -> allocate -> extract -> place -> set/clear mask -> replace
                                  |          |
                                  +-> native guide
extract -> append
split   -> reversible suffix -> append
```

None loads a model, creates a prompt, selects a sampler, samples, decodes, or
claims a visual objective.

`noise_mask` follows current official H3 sampler semantics: video and
structural-audio masks remain separate tensors, and a value of `1` requests
generation while `0` preserves the supplied latent token. CAUCE evaluates
linear, smoothstep, or smootherstep temporal ramps at each stream's own token
centers. It does not collapse 24 fps frames and 40 Hz audio tokens onto one
approximate clock.

## Dependency policy

Core mathematics remains importable without ComfyUI. PyTorch and ComfyUI are
lazy-imported only at tensor/runtime boundaries. The package adds no pip
dependencies and never owns the GPU software stack.
