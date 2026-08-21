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
2. A copied joint AV context additionally needs an integer 40 Hz audio
   endpoint. The shared sequence is `39+51m` frames.
3. A continuation parent retains H3's causal origin. CAUCE crops only its
   post-accept tail; it never drops a phase-shifting latent prefix.
4. Independent H3 latents are decoded independently. CAUCE does not concatenate
   them and assume the visual VAE phase restarts at each join.
5. Masked continuation is the baseline. Combining the same content as native
   guide conditioning is exposed only as an explicit experiment.
6. Master audio is placed and delivered on the rational/sample clock and can
   remain authoritative instead of being reconstructed through AudioVAE.
7. Dense official H3 remains the baseline for the laboratory 5090. Sparse
   attention, caches, turbo LoRAs, and third-party quantizations are candidates,
   not hidden dependencies.

## Compatibility policy

The H3 adapter imports official classes lazily and fails closed if their public
runtime signatures disappear. Before promoting a new release, repeat the
laboratory matrix in `VALIDATION.md` against the exact Comfy commit recorded in
the resulting receipts.
