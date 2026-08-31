# Native H3 workflow recipes

These are graph recipes, not bundled workflow JSON. Resolve exact official
sockets against live `/object_info` before creating a UI graph or API prompt.

## Visible H3 preflight

The planning nodes expose official temporal preprocessing rules without
replacing the official encoder or conditioning node:

```text
requested frames / dimensions
  -> CauceH3ResolveTargetShape
  -> resolved frame count ----------------> official H3 length

decoded guide clip + official target latent
  -> CauceH3PrepareGuideClip
  -> accepted IMAGE + resolved frame index -> MiniMaxH3AddGuide

decoded reference clip + requested target frames
  -> CauceH3PrepareReferenceClip
  -> accepted IMAGE -----------------------> MiniMaxH3ReferenceToVideo
```

Place `CauceH3InspectConditioning` on the active positive-conditioning edge
immediately before the guider when a run needs structural evidence. Its
`positive` output is the unmodified input.

For cumulative native-state branching:

```text
cumulative origin-zero AV latent
  -> CauceH3SplitAVLatent(cut=legal 17k+5 prefix)
      -> prefix LATENT     -> alternate continuation
      -> suffix AV SPAN    -> append to reconstruct the original state
```

## Native AV tail continuation

The operation is composed from low-level CAUCE data transformations and ordinary
official H3 inference:

```text
previous AV latent
  -> CauceH3PlanAVWindow(overlap=22, extension=119)
      -> CauceH3AllocateAVWindow ---------------------------> sampler latent
      -> window_frames ------------------------------------> H3 conditioning length

previous AV latent
  -> CauceH3ExtractAVSpan(start=previous_frames-overlap, count=overlap)
      -> either:
         A. CauceH3AddAVSpanGuide(target_frame_idx=0)
         B. CauceH3PlaceAVSpan(target_frame_idx=0)
            -> CauceH3SetAVDenoiseInterval(
                 start=overlap, count=extension,
                 inside=1, outside=0
               )

official MiniMaxH3ImageToVideo positive conditioning
  -> CauceH3AddAVSpanGuide
  -> official guider / sampler using allocated target
  -> CauceH3ExtractAVSpan(
       timeline_origin=window_start,
       start=overlap,
       count=extension
     )
  -> CauceH3AppendAVSpan(base=previous AV latent)
```

The allocated target, positive conditioning, sampler, scheduler, seed, model,
and decode remain separate visible graph edges. The canonical graph follows
the current official template and connects the loaded model directly to the
guider and scheduler. `MiniMaxH3SigmaShift` remains available as an opt-in
native experiment, but it must not silently enter every CAUCE operation. The
first characterized comparison should retain the upstream reference geometry:

```text
overlap            22 frames
new suffix        119 frames
sampled window    141 frames
```

The keyframe-overlap variant follows an inspected, known-running MIT reference.
The masked-overlap variant follows current official H3 per-token mask support.
Both must earn separate live schema, execution, and visual evidence.

## Native AV completion and replacement

Completion always operates on one complete H3 target lattice. Known spans are
placed first; the unknown interval receives denoise strength `1`, and preserved
context receives `0`. Video and structural-audio masks are built independently
at their native token rates.

```text
target AV lattice
  -> CauceH3PlaceAVSpan(left known span, exact target frame)
  -> CauceH3PlaceAVSpan(right known span, exact target frame)
  -> CauceH3SetAVDenoiseInterval(
       start=unknown_start,
       count=unknown_frames,
       inside_video/audio=1,
       outside_video/audio=0,
       fade_in/out=explicit
     )
  -> ordinary official H3 conditioning / guider / sampler
  -> CauceH3ExtractAVSpan(unknown interval)
  -> CauceH3ClearAVDenoiseMask
```

This supports four distinct graph topologies without hiding intent in a node:

- backward prefix: place known future context on the right and generate before
  it; rebase only if visual and 40 Hz phases align;
- two-sided infill: preserve native spans on both sides and generate the middle;
- local replacement: regenerate an interval, extract it, and replace that same
  interval in cumulative state;
- two-source connection: place compatible native spans from different sources
  around an unknown interval.

Hard boundaries use zero fades. Continuous boundaries use linear,
`smoothstep`, or `smootherstep` ramps evaluated at token centers. A softer mask
is a sampling hypothesis, not guaranteed visual continuity, so retain exact
parameters in the run receipt and compare it empirically against the hard mask.

## Native rollback and branching

```text
cumulative AV state
  -> CauceH3SplitAVLatent(cut_frame)
      -> prefix native AV state -> persist as branch checkpoint
      -> suffix AV span         -> retain for exact reconstruction
```

The suffix can be appended to reconstruct the original state or replaced by a
new continuation. This is deterministic state management and requires no H3
sampling.

## Spatial and spatiotemporal masked editing

`CauceH3ApplyVideoDenoiseMask` accepts either one static `MASK` or one mask per
decoded frame in an exact interval. It resizes continuous spatial values to the
visual latent grid and reduces decoded frames covered by each H3 visual token
with `amax`.

```text
source native AV state
  -> optional CauceH3SetAVDenoiseInterval
  -> CauceH3ApplyVideoDenoiseMask(
       mask=static or per-frame,
       combine=replace or multiply,
       audio_strength=0
     )
  -> ordinary official conditioning / guider / sampler
  -> CauceH3ClearAVDenoiseMask
```

Use `replace` for a standalone spatial edit. Use a temporal interval followed
by `multiply` for a local retake: the resulting generation volume is the
intersection of temporal and spatial selections. The current official core
pools to the 2x2 visual patch grid and bounds fractional token levels; evaluate
edge behavior empirically at the final resolution.

## Native video outpainting

```text
source native AV state
  -> CauceH3ExpandAVCanvas(
       target width/height = multiples of 32,
       offset x/y = multiples of 32,
       source strength = 0,
       new-region strength = 1,
       audio strength = 0
     )
  -> ordinary official H3 conditioning / guider / sampler
  -> CauceH3ClearAVDenoiseMask
```

The source visual tensor is copied without interpolation. The structural-audio
stream and duration remain unchanged. Centered and explicit-offset placements
are separate topology variants because their bound offsets differ.

## Bounded second-pass refinement

Refinement is masked V2V over the original native state, not an independent
generation and not a custom sampler:

```text
source native AV state
  -> CauceH3SetAVDenoiseInterval(
       start=0,
       count=complete duration,
       inside video strength=experimental bounded value,
       inside/outside audio strength=0
     )
  -> optional CauceH3ApplyVideoDenoiseMask(combine=multiply)
  -> ordinary official H3 conditioning / guider / sampler
  -> CauceH3ClearAVDenoiseMask
```

No default strength is described as visually accepted. Characterize a ladder
on one source, prompt, and seed before choosing a production range.

## Workflow-only H3 patterns

These need no new CAUCE node or semantic operation:

- structured multishot prompting is a prompt protocol inside an ordinary
  `generate.keyframed` or `generate.from_references` graph;
- window-aligned motion references bind a different exact reference-video
  range to each prebound generation or continuation graph in a serial plan;
- first, last, and multiple interior image or clip anchors are explicit chains
  of official `MiniMaxH3AddGuide` nodes.

The production layer owns per-window media bindings. Runtime Control executes
already bound graphs and does not infer window correspondence.

## Fixed soundtrack alignment

Keep the final soundtrack as the editorial clock. Convert timeslots to exact 24
fps frame ranges and mux the unchanged soundtrack after visual assembly. H3's
structural-audio latent remains synchronized internally wherever native AV
continuation requires it, but the production soundtrack is not a model input.

## Sources

- [Official MiniMax H3 repository](https://github.com/MiniMax-AI/MiniMax-H3)
- [Official ComfyUI H3 nodes](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py)
- [Official MiniMaxH3AddGuide documentation](https://github.com/Comfy-Org/embedded-docs/blob/main/comfyui_embedded_docs/docs/MiniMaxH3AddGuide/en.md)
- [Official AddGuide implementation PR](https://github.com/Comfy-Org/ComfyUI/pull/15439)
- [Official continuous per-token mask implementation PR](https://github.com/Comfy-Org/ComfyUI/pull/15375)
- [Inspected native continuation reference](https://github.com/ttulttul/ComfyUI-Minimax-H3-Continuation)
- [Community masked V2V and outpaint workflows](https://github.com/ckinpdx/ComfyUI-MMH3Tools)
- [Community H3 refinement workflows](https://github.com/mdkberry/comfyui_workflows/tree/main/workflows_by_model/Minimax-H3)
