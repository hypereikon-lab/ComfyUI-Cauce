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

The standard masked-denoise Confluence graph passed tensor/runtime validation
but failed on heterogeneous real gesture material. This rejects the sampler as
a production method; it does not reject the 124-frame causal domain, exact
splice geometry, or fixed-duration acceptance contract.

Sources reviewed for the replacement:

- [LanPaint paper](https://arxiv.org/abs/2502.03491): training-free partial
  conditional sampling through bidirectional guided Langevin dynamics, without
  backpropagation or model training.
- [LanPaint ComfyUI implementation](https://github.com/scraed/LanPaint): v2.1.0
  fixes current H3 support and exposes a custom advanced sampler compatible
  with H3 nested latents. It is GPL-3.0 and remains separately installed.
- [Current ComfyUI sampler path](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/sample.py):
  native `noise_mask` values are passed into the custom sampler rather than
  being compiled as binary UI state.

The replacement keeps three different mathematical objects:

```text
sampling strength s(t,x,y) ∈ [0,1]
hard accepted interval       a(t) ∈ {0,1}
decoded output opacity       o(t,x,y) ∈ [0,1]
```

These fields must not be collapsed into one mask. `s` controls the conditional
sampler, `a` controls what may enter the production result, and `o` composites
the accepted decoded proposal with the two source clips. The default `s` and
`o` are raised cosines, while either can be replaced by arbitrary Comfy `MASK`
data. H3 audio remains a zero-masked structural stream and is discarded; the
fixed master soundtrack is not encoded or conditioned through this path.

CAUCE does not copy LanPaint code. Workflow 60 depends on its public node
interface, while CAUCE independently implements the temporal geometry, field
construction, causal-token projection, acceptance, and splice.

## Compatibility policy

The H3 adapter imports official classes lazily and fails closed if their public
runtime signatures disappear. Before promoting a new release, repeat the
laboratory matrix in `VALIDATION.md` against the exact Comfy commit recorded in
the resulting receipts.
