# Native H3 workflow recipes

These are graph contracts, not bundled workflow JSON. Resolve exact sockets
from the live `/object_info` registry before materializing a browser graph or
API prompt.

## 1. Two-sided guide window

Purpose: generate a new interval conditioned by the outgoing gesture of video A
and incoming gesture of video B.

```text
A IMAGE batch ─┐
               ├─ CaucePrepareH3TwoSidedGuideWindow
B IMAGE batch ─┘       | left_guide, frame_idx = 0
                       | right_guide, frame_idx = returned value
                       | target_frames
                       v
official MiniMaxH3ImageToVideo (fresh target, no first/last image)
  -> official MiniMaxH3AddGuide(left_guide, 0)
  -> official MiniMaxH3AddGuide(right_guide, right_guide_frame_idx)
  -> official guider / sampler / H3 decode
  -> CauceAssembleH3TwoSidedGuideWindow(original A, original B, decoded target, plan)
  -> Save Video
```

The fresh target matters: the two guide clips are the explicit conditioning
anchors. CAUCE accepts only the center between them, so guide reconstruction is
not duplicated in the final edit.

Default frame geometry:

```text
0                    22                           102                  124
|-- A tail guide -----|------- generated center ---|-- B head guide ----|
                                              accepted: [22, 102)
```

The graph uses a documented official H3 conditioning operation, but the
generated range remains subject to visual evaluation. Start with 22-frame guides;
compare only one variable at a time when testing 39 or 56 frames.

Prompting should describe the intended transition motion without narrating the
guide mechanics. Hold prompt, seed, model, resolution, sampler, and steps fixed
when comparing guide lengths.

## 2. Native tail continuation

For extending one H3 generation, compose the external
`ComfyUI-Minimax-H3-Continuation` pack rather than duplicating it inside CAUCE.
Its characterized workflow keeps a native tail guide, builds a fresh official
target, supplies the exact guide through the supported conditioning path,
discards the regenerated overlap, and appends the new suffix.

Use the external pack's documented 22-frame overlap and live node schemas. Save
the resulting packed latent with `CauceSaveAVLatent` only when persistence is
useful to the production graph.

## 3. Motion-reference media

CAUCE motion maps act before H3:

```text
source IMAGE batch or generated primitive frames
  -> one or more CAUCE map builders
  -> CauceComposeMotionMaps
  -> CauceWarpImage
  -> video-reference encoder
  -> official Ref2VA or MiniMaxH3AddGuide graph
```

This path gives the model observable reference media. It does not assume that a
geometric operation in pixel space has a corresponding valid intervention in
H3's internal state.

## 4. Fixed soundtrack alignment

Keep the final soundtrack as the editorial clock. Convert its time slots into
24 fps frame ranges, generate the required visual intervals, and mux
the unchanged soundtrack only after the visual edit is assembled. Do not add
audio conditioning unless a specific H3 workflow genuinely needs it.

## Sources

- [Official MiniMax H3 repository](https://github.com/MiniMax-AI/MiniMax-H3)
- [Official ComfyUI H3 nodes](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py)
- [Official MiniMaxH3AddGuide documentation](https://github.com/Comfy-Org/embedded-docs/blob/main/comfyui_embedded_docs/docs/MiniMaxH3AddGuide/en.md)
- [Native H3 continuation pack](https://github.com/ttulttul/ComfyUI-Minimax-H3-Continuation)
