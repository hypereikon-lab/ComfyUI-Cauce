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
  -> CauceH3ExtractAVSpan(start=previous_frames-22, count=22)
      -> CauceH3AddAVSpanGuide(target_frame_idx=0)

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
sigma shifts, and decode remain separate visible graph edges. The first
characterized comparison should retain the upstream reference geometry:

```text
overlap            22 frames
new suffix        119 frames
sampled window    141 frames
```

This mechanism is derived from the inspected, known-running MIT reference
implementation, but CAUCE's implementation must earn live schema, execution,
and visual evidence independently.

## Two-sided decoded guide window

No CAUCE node owns this workflow. Compose it from one range primitive and
official/vanilla nodes:

```text
A frames -> CauceAcceptDecodedRange(last guide range) -> AddGuide at 0
B frames -> CauceAcceptDecodedRange(first guide range) -> AddGuide at target-guide

official H3 fresh target -> official sampling -> decoded target
decoded target -> CauceAcceptDecodedRange(guide, target-2*guide)

A complete -> vanilla ImageBatch -> accepted generated range
           -> vanilla ImageBatch -> B complete
```

For a 124-frame target with 22-frame guides, accept `[22, 102)`, or 80
generated frames. Those numbers belong to the workflow/run receipt rather than
to hidden node state.

## Motion-reference media

```text
source IMAGE batch or generated primitive frames
  -> CAUCE map builders
  -> CauceComposeMotionMaps
  -> CauceWarpImage
  -> inspect decoded reference
  -> official Ref2VA or MiniMaxH3AddGuide graph
```

This path does not modify H3 latents or sampler internals. A correct coordinate
map is not evidence that H3 followed it; evaluate that separately.

## Fixed soundtrack alignment

Keep the final soundtrack as the editorial clock. Convert timeslots to exact 24
fps frame ranges and mux the unchanged soundtrack after visual assembly. H3's
structural-audio latent remains synchronized internally wherever native AV
continuation requires it, but the production soundtrack is not a model input.

## Sources

- [Official MiniMax H3 repository](https://github.com/MiniMax-AI/MiniMax-H3)
- [Official ComfyUI H3 nodes](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py)
- [Official MiniMaxH3AddGuide documentation](https://github.com/Comfy-Org/embedded-docs/blob/main/comfyui_embedded_docs/docs/MiniMaxH3AddGuide/en.md)
- [Inspected native continuation reference](https://github.com/ttulttul/ComfyUI-Minimax-H3-Continuation)
