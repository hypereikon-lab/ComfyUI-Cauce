# Semantic operations

A CAUCE operation is a typed graph-level function over decoded media or native
H3 state. Operations are orthogonal and composable; their names do not imply a
production sequence.

The catalog groups them into three non-sequential families. This grouping is
machine-readable in `operations/catalog.json` and does not change any operation
id or individual contract hash.

```text
H3 conditioning grammar
  generate.keyframed
  generate.from_references
  generate.with_guides

native H3 AV state algebra
  continue.native_av
  complete.native_av
  rollback.native_av

decoded media algebra
  frames.assemble
```

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
cauce            deterministic range, AV-state, or persistence primitive
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
| `continue.native_av` | extend synchronized packed H3 AV state through keyframe or masked overlap | official H3 + CAUCE AV primitives | contract + offline topology | keyframe path executes synthetically; masked paths unit-validated |
| `complete.native_av` | generate a prefix/interior/replacement while preserving explicit native AV context | official H3 + CAUCE placement/mask/replace primitives | contract + offline topology | deterministic layer unit-validated |
| `rollback.native_av` | split cumulative AV state into a branchable prefix and reversible suffix | CAUCE split/persistence primitives | contract + offline topology | unit-validated |
| `frames.assemble` | select and concatenate exact decoded ranges | CAUCE ranges + vanilla assembly | contract + offline topology | deterministic layer unit-validated |

The first family describes supported ways to construct H3 conditioning. The
second treats packed synchronized AV state as an explicit value that can be
extended, temporally completed, split, persisted, and recomposed. The third is
deterministic post-decode work. See [Operation model](OPERATION_MODEL.md) for
the lifecycle from primitive through run evidence.

“Contract only” means no paired, import-tested UI graph and executable API
template are shipped yet. It does not prevent the graph from being composed;
it prevents the repository from claiming a reusable materialized artifact.
The offline topology is a checked design dossier, not an additional artifact
state and not an executable workflow. See [Topology drafts](TOPOLOGY_DRAFTS.md).

## Composability

Operations can be connected by their typed outputs rather than by catalog
order. Examples:

```text
generate.keyframed
  -> continue.native_av
  -> continue.native_av
```

```text
decoded sources
  -> generate.from_references
  -> continue.native_av
```

```text
native AV state + known left/right spans
  -> complete.native_av
  -> rollback.native_av
  -> alternate continue.native_av
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

The catalog now describes native temporal completion through current official
H3 per-token mask semantics, but no masked sampling topology is yet claimed as
executed or visually accepted. It does not claim automatic intent-to-graph
synthesis, arbitrary sampler modification, generative audio, training,
acceleration, or streaming. A new operation enters the catalog only with a
typed data contract, explicit node ownership, and honest artifact and evidence
state.
