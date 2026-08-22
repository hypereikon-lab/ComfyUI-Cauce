# Research basis and compatibility ledger

CAUCE is an independent MIT implementation. It does not copy code from the
GPL node packs studied during design. This ledger records the upstream behavior
and community experiments used to challenge the architecture.

## Pinned sources reviewed on 2026-08-21

- [MiniMax H3 official repository](https://github.com/MiniMax-AI/MiniMax-H3):
  model architecture, supported FL2VA/Ref2VA inputs, 24 fps output, 40 Hz audio
  latent rate, causal visual VAE, and release limitations.
- [ComfyUI native H3 nodes at `28864ca`](https://github.com/Comfy-Org/ComfyUI/blob/28864ca8fed67cc05a957710f88ce23aff75fd2f/comfy_extras/nodes_minimax_h3.py):
  official FL2VA, Ref2VA, AddGuide, nested AV latent construction, frame grid,
  reference ordering, and runtime signatures called by CAUCE.
- [ComfyUI H3 model integration at `28864ca`](https://github.com/Comfy-Org/ComfyUI/blob/28864ca8fed67cc05a957710f88ce23aff75fd2f/comfy/ldm/minimax/model.py):
  packed stream layout, temporal support pattern, and per-stream mask behavior.
- [Official ComfyUI H3 templates at `e95e3b2`](https://github.com/Comfy-Org/workflow_templates/tree/e95e3b20567bea8df16510c8390b7f897b7e6d4b/templates):
  current loader, sampling, decoding, and output graph conventions.
- [H3 Motion Context / latent-masking experiments at `87de57b`](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef/tree/87de57ba619297503fa49c9594c0c021d5b0c261):
  independent community evidence around long-form continuation, shared AV
  boundaries, two-ended bridges, audio feathers, causal decode, persistence,
  and bounded final assembly. Its GPL code is not included in CAUCE.
- [Comfy-Org MiniMax-H3 model files](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main):
  exact paths, byte sizes, and LFS SHA-256 values used by read-only preflight.

## Decisions produced by the comparison

1. H3 video-valid runs are `17k+5`; visual latent length is `5k+2`.
2. The first implementation copied a joint AV context and therefore required
   `39+51m` frames. Production now copies video only and accepts every legal
   visual boundary `5+17m`; H3 audio rows remain frozen scaffolding.
3. A continuation parent retains H3's causal origin. CAUCE crops only its
   post-accept tail; it never drops a phase-shifting latent prefix.
4. Independent H3 latents are decoded independently. CAUCE does not concatenate
   them and assume the visual VAE phase restarts at each join.
5. Masked continuation is the baseline. Combining the same content as native
   guide conditioning is exposed only as an explicit experiment.
6. Master audio is placed and delivered on the rational/sample clock and can
   remain authoritative instead of being reconstructed through AudioVAE.
7. Dense official H3 remains the baseline for the laboratory 5090. Acceleration,
   streaming, LoRA training, and generated audio are outside the production
   scope rather than hidden dependencies or roadmap items.

## Confluence review — 2026-08-22

The v1–v3 graph passed tensor/runtime validation but failed on heterogeneous
real gestures. The audit found that the failure preceded sampler choice:

1. The requested one-second center had been interpreted as one second per side.
2. LanPaint overscan enlarged the actual unknown interval to 72 frames/3 s.
3. The laboratory core, ComfyUI v0.33.1 (`72865f4`), predates the official H3
   per-token mask path merged in PR #15375 and `MiniMaxH3AddGuide` from PR
   #15439. A nested mask reached the sampler, but preserved H3 rows did not yet
   receive their own conditioning timestep and latent injection.

Current official ComfyUI provides the missing semantics:

- [`MiniMaxH3AddGuide`](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py)
  anchors an image or a valid `17k+5` clip at an arbitrary frame index.
- [`MiniMaxH3._denoise_mask_conds` and `scale_latent_inpaint`](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/model_base.py)
  preserve source latents and expose video/audio masks to the model.
- [MiniMax H3 model rows](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/model.py)
  use `t_row = clamp(1 - m·sigma, max=t_pin)`, so generated and preserved rows
  are labeled at different noise levels.

The older ecosystems support the same architectural conclusion without being
copied into CAUCE:

- [VACE](https://github.com/ali-vilab/VACE/blob/main/vace/models/wan/wan_vace.py)
  does not treat a mask as sufficient by itself. It concatenates encoded source
  video and mask into a dedicated control latent; preserved visual evidence is
  an explicit model input.
- [AnimateDiff](https://github.com/guoyww/AnimateDiff/blob/main/animatediff/models/motion_module.py)
  applies attention along the frame axis with temporal positions. Its evolved
  context schedulers overlap windows and weight their centers more strongly,
  motivating local context adjacent to a seam rather than distant endpoints.
- [ProPainter](https://github.com/sczhou/ProPainter/blob/main/inference_propainter.py)
  propagates information bidirectionally with completed optical flow, then
  combines local neighboring frames and global references. CAUCE does not add
  those dependencies, but mirrors the information topology with left and right
  guide clips.
- [LanPaint](https://github.com/scraed/LanPaint) remains a useful training-free
  inpainting sampler, but it cannot supply H3 model semantics missing from an
  older core. It is no longer a workflow-60 dependency.

Confluence v4 keeps three mathematical objects separate:

```text
model generation support  s(t,x,y) ∈ {0,1}
accepted decoded interval          a(t) ∈ {0,1}
decoded splice opacity             o(t,x,y) ∈ [0,1]
```

For the default 124-frame domain with cut `c=62`, CAUCE chooses token
boundaries `l=51`, `r=73` minimizing `|(r-l)-24|` under `c-l=r-c`. The unknown
middle is therefore 22 frames. Valid guide clips are
`G_L=[29,51)` and `G_R=[73,95)`. The standard H3 sampler operates only on
`[51,73)`; a four-frame raised cosine is used only after decode. H3 audio
remains zero-masked structural scaffolding and is discarded.

## Compatibility policy

The H3 adapter imports official classes lazily and fails closed if their public
runtime signatures disappear. Before promoting a new release, repeat the
laboratory matrix in `VALIDATION.md` against the exact Comfy commit recorded in
the resulting receipts.
