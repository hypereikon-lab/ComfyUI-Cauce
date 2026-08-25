# Semantic operations

A CAUCE operation is a typed graph-level function over decoded media or native
H3 state. Operations are orthogonal and composable; their names do not imply a
production sequence.

```text
inputs + explicit parameters
  -> official H3 / vanilla ComfyUI / CAUCE primitive graph
  -> outputs + retained state + run evidence
```

CAUCE may own the operation contract while owning only some, or none, of the
nodes that implement it. Every graph stage declares one of these owners:

```text
official-comfy   model-specific H3 conditioning
vanilla-comfy    loaders, sampler, decode, batching, and file outputs
cauce            deterministic range, AV-state, map, or persistence primitive
```

The H3 planning and conditioning-inspection nodes are optional graph preflight
primitives. They make official temporal rules and active metadata visible, but
do not create a separate semantic operation or replace official H3 nodes.

## Current catalog

| Operation | Principal data function | Implementation | Artifact state | Evidence |
| --- | --- | --- | --- | --- |
| `generate.keyframed` | generate from prompt and optional endpoint frames | official H3 / vanilla | contract + offline topology | defined |
| `generate.from_references` | generate from ordered reference images or clips | official H3 / vanilla | contract + offline topology | defined |
| `generate.with_guides` | place decoded guides at exact target-frame indices | official H3 / vanilla | contract + offline topology | defined |
| `continue.native_av` | extend synchronized packed H3 AV state | official H3 + CAUCE AV primitives | contract + offline topology | executes synthetically |
| `connect.two_sided_guides` | generate a center conditioned by both decoded sides | official H3 + CAUCE ranges + vanilla assembly | contract + offline topology | defined |
| `reference.transform` | construct decoded reference media from coordinate maps | CAUCE maps, optionally followed by official H3 | contract + offline topology | deterministic layer unit-validated |
| `frames.assemble` | select and concatenate exact decoded ranges | CAUCE ranges + vanilla assembly | contract + offline topology | deterministic layer unit-validated |

“Contract only” means no paired, import-tested UI graph and executable API
template are shipped yet. It does not prevent the graph from being composed;
it prevents the repository from claiming a reusable materialized artifact.
The offline topology is a checked design dossier, not an additional artifact
state and not an executable workflow. See [Topology drafts](TOPOLOGY_DRAFTS.md).

## Composability

Operations can be connected by their typed outputs rather than by catalog
order. Examples:

```text
reference.transform
  -> generate.from_references
  -> continue.native_av
  -> frames.assemble
```

```text
generate.keyframed
  -> continue.native_av
  -> continue.native_av
```

```text
decoded sources
  -> connect.two_sided_guides
  -> frames.assemble
```

The cumulative native AV latent is an explicit output and input. Persist it
whenever future continuation is expected; an MP4 alone cannot recover the same
native state.

## Materialization rule

Each operation can eventually retain one or more variant-scoped pairs:

```text
<operation>.<variant>.ui.json
<operation>.<variant>.api.template.json
```

The pair must come from the same active ComfyUI graph, be validated against the
same live `/object_info` capture, and carry independent hashes. Optional graph
branches use separate variants rather than muted or bypassed nodes. A paired
variant does not imply that the operation's other variants are materialized.

Until those conditions hold, `artifacts.state` remains `contract-only` and
`artifacts.pairs` remains empty. Once at least one exact variant has a valid
pair, the state becomes `paired-graphs` and every retained pair records its
variant id, paths, and canonical JSON hashes.

## Evidence model

Artifact state and behavioral evidence are independent:

```text
artifact state
  contract-only | paired-graphs

evidence level
  defined | unit-validated | schema-validated | executes | visually-characterized

visual verdict
  unassessed | accepted | rejected | mixed
```

An operation can execute without having a reusable retained graph, as occurred
with the synthetic `continue.native_av` smoke. Conversely, a materialized graph
can be schema-valid while its visual objective remains unassessed.

## Current exclusions

The catalog does not currently claim masked temporal inpainting, automatic
intent-to-graph synthesis, arbitrary sampler modification, generative audio,
training, acceleration, or streaming. A new operation enters the catalog only
with a typed data contract, explicit node ownership, and honest artifact and
evidence state.
