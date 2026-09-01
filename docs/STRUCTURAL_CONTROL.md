# Official MiniMax H3 structural control

`generate.with_control` is a graph contract around official ComfyUI and the
official H3 Fun Control union model patch. CAUCE does not implement a second
ControlNet, wrap the sampler, or classify the meaning of the control video.

## Canonical graph

```text
prompt + target geometry
  -> MiniMaxH3ImageToVideo

precomputed structural video
  -> GetVideoComponents
  -> CauceH3PlanControlClip          # read-only report

H3 Base + union model patch + VAE
  -> MiniMaxH3FunControlNetApply

conditioning + target AV latent
  -> CauceH3InspectPackedSequence    # read-only report
  -> BasicGuider / BasicScheduler / SamplerCustomAdvanced
  -> VAEDecode
```

The structural video may be Canny, depth, pose, HED, MLSD, or an arbitrary
precomputed signal the checkpoint can interpret. Producing that signal is an
upstream workflow concern. CAUCE does not need a separate semantic entity for
each preprocessor.

## Exact current core behavior

For a target of `N` decoded frames, the official control patch fits its input
to exactly `N` frames:

```text
source longer than N  -> truncate the tail
source shorter than N -> repeat the final source frame
same length           -> identity in time
```

Spatial input is resized bilinearly and center-cropped to target H3 VAE
geometry. In mask mode, current core thresholds the mask at `> 0.5`, repeats or
truncates it to the target duration, and reads `source_video` behind the mask.
Therefore a smooth mask supplied to this official path is not presently a
continuous denoise field; that is different from CAUCE native latent masks.

## Packed sequence observability

`CauceH3InspectPackedSequence` mirrors the row accounting in official
`PackedLayout`:

```text
text
+ keyframe visual/audio rows
+ reference image/video/audio rows
+ target structural-audio rows
+ target visual patch rows
```

Target visual rows are:

```text
video_tokens * ceil(latent_height / 2) * ceil(latent_width / 2)
```

Target audio rows are `2 * audio_tokens`. Scheduled conditioning entries can
have different text/reference lengths, so the inspector reports every entry
and selects the largest sequence. The row count is exact for the captured core
contract. `estimated_bytes_per_row` is a separate linear calibration (default
151,000 bytes) and must never be treated as a universal VRAM guarantee.

## Compatibility gates

The laboratory runtime may be older than merged core support. Before install:

1. capture `/object_info` and require `MiniMaxH3FunControlNetApply`,
   `ModelPatchLoader`, and the two CAUCE inspection nodes;
2. pin the ComfyUI commit and model-patch file hash;
3. verify reference/control compatibility against ComfyUI #16020 or an
   equivalent merged implementation;
4. verify denoise-mask velocity against #15988 and live mask behavior against
   #15978/#15981;
5. execute at native pixel bounds and record packed rows, peak VRAM/RAM,
   runtime, seed, strength, start/end percentages, and every input hash.

Reference+control is declared but has no topology dossier yet because current
merged core rejects `minimax_refs` and `minimax_keyframes` in the control path.
That absence is intentional evidence hygiene, not an unfinished hidden graph.

## First empirical matrix

At one fixed seed and native geometry:

```text
control kind       pose / depth / canny
strength           0.6 / 0.8 / 1.0
end_percent        0.6 / 0.8 / 1.0
prompt             neutral continuity / intended movement
```

Judge structure retention, texture recovery after control release, prompt
adherence, temporal stability, runtime, and memory separately. Community
reports suggest pose can preserve figures better than depth, values above 1
can degrade, and releasing control before the final denoising steps may restore
texture. Those are hypotheses for the matrix, not accepted defaults.

## Primary sources

- [Official ComfyUI H3 Fun Control integration](https://github.com/Comfy-Org/ComfyUI/pull/15975)
- [MiniMax H3 Fun ControlNet Union model card](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union)
- [Reference/control and dynamic-VRAM fix](https://github.com/Comfy-Org/ComfyUI/pull/16020)
- [Mask velocity correction](https://github.com/Comfy-Org/ComfyUI/pull/15988)
- [DiffSynth-Studio H3 ControlNet support](https://github.com/modelscope/DiffSynth-Studio/commit/013296ed313f22ecc78b1c27df75275bb9139a9c)
