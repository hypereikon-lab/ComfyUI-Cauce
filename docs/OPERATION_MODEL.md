# Operation model

CAUCE exposes ten orthogonal graph-level operations in three capability
families. The families are a classification, not an execution order.

## H3 conditioning grammar

```text
generate.keyframed
generate.from_references
generate.with_guides
```

These operations describe official H3 conditioning configurations: prompt with
optional endpoints, ordered Ref2VA media, and exact-frame AddGuide chains.
Official ComfyUI owns encoding, conditioning, model inference, and decode.
CAUCE owns the typed portable contract and optional preflight primitives; it
does not claim to have implemented the official H3 capability.

## Native H3 AV state algebra

```text
continue.native_av
complete.native_av
edit.masked_video
reframe.outpaint_video
refine.video
rollback.native_av
```

These operations treat synchronized packed video and structural-audio latent
state as a first-class artifact:

```text
state -- extend --------------------------> longer state
state -- complete[start, end) ------------> completed/replaced state
state + mask -- edit[x, y, t] ------------> selectively edited state
state -- expand canvas -------------------> reframed/outpaint target state
state -- bounded second pass -------------> refined state
state -- split[cut] ----------------------> prefix + reversible suffix
```

`complete.native_av` is the general temporal-latent-inpainting contract. Its
backward-prefix, two-sided-infill, local-replacement, and two-source-connection
variants differ by placement of known spans and the unknown interval; they are
not separate hidden mechanisms.

`edit.masked_video` owns arbitrary static or animated spatial selection.
`reframe.outpaint_video` owns aligned visual-lattice expansion.
`refine.video` reuses the same official mask path at bounded strength; it is
not a separate sampler.

## Decoded media algebra

```text
frames.assemble
```

This family performs exact deterministic work after decode. It neither samples
H3 nor substitutes decoded video for retained native state.

## Lifecycle

The repository uses these terms strictly:

```text
primitive
  one low-level node/data transform

operation
  one typed graph-level data function

variant
  one explicit static topology of an operation

workflow pair
  UI graph + API template exported from the same live graph

invocation
  workflow pair + concrete media and parameter bindings

run
  exact prompt id + runtime manifest + artifacts + immutable receipt

evidence
  technical state and separate human visual verdict
```

A topology dossier is not a workflow pair. A schema-valid workflow is not an
execution. A completed queue item is not visual acceptance. Each transition is
recorded separately and may fail without promoting the next state.

## Composition

Operations connect through explicit typed outputs. Their catalog order has no
semantic meaning.

```text
generate.keyframed
  -> native AV state
  -> continue.native_av
  -> native AV state
  -> rollback.native_av
```

```text
known native spans
  -> complete.native_av
  -> completed native AV state
```

```text
accepted decoded ranges
  -> frames.assemble
```

Prompt text and media remain opaque inputs. Workflow graphs assign production
meaning; CAUCE does not infer objects, actions, scenes, or editorial intent.
