# Workflow contracts

These are the first reproducible graphs to construct against the live ComfyUI
`/object_info` registry. They intentionally share the same source material,
prompt, seed, model, sampler, and output resolution.

## W0 — decoded join control

```text
left frames  ─┐
              ├─ concatenate ─ video save
right frames ─┘
```

Purpose: expose the original cut and provide the exact duration/reference
sequence. No H3 inference.

## W1 — masked temporal inpainting

```text
left frames ─┐
             ├─ CauceBuildSeamWindow ─ working images ─ H3 VAE encode ─┐
right frames ┘                                                          │
                                                                        ├─ CaucePrepareH3TemporalInpaint
MiniMaxH3ImageToVideo(prompt, no first/last image) ─ target AV latent ──┘

masked AV latent ─ sampler ─ H3 VAE decode ─ CauceApplySeamPatch ─ video save
```

Purpose: test ComfyUI's native per-row H3 mask without duplicate guide
conditioning.

## W2 — bidirectionally guided temporal inpainting

W2 is W1 plus two official guide nodes:

```text
working images ─ slice G_L ─ MiniMaxH3AddGuide(frame_idx = G_L.start) ─┐
                                                                      ├─ positive conditioning
working images ─ slice G_R ─ MiniMaxH3AddGuide(frame_idx = G_R.start) ─┘
```

The mask path is unchanged. Purpose: measure whether explicit incoming and
outgoing motion clips improve the seam beyond the preserved main latent.

## W3 — continuous temporal denoise strength

W3 is W2 with only the sampling field changed:

```text
CauceBuildTemporalDenoiseField(shoulder_tokens = 3, curve = cosine)
  └─ denoise_strength ─> CaucePrepareH3TemporalInpaint(
                           mask_mode = continuous,
                           continuous_projection = mean)
```

Purpose: test whether partial denoising at the incoming and outgoing token rows
reduces boundary acceleration or resets relative to W2's binary field. Guides,
prompt, seed, source, sampler, accepted range, and decoded opacity are identical.

An animated spatial-mask variant is composed without another CAUCE node:

```text
temporal denoise_strength ─┐
                           ├─ Combine Masks(multiply) ─ generation_support
animated Comfy MASK[T,H,W] ┘
```

Run this only after the temporal-only W3 comparison.

## W4 — native-latent seam research

```text
left H3 AV latent  ─┐
target H3 AV latent ├─ CaucePrepareH3NativeLatentInpaint ─ sampler ─ decode
right H3 AV latent ─┘
```

Purpose: test whether source-native latent context improves continuity. This is
not a production dependency and must be compared with W2 at matched settings.

## Shared starting configuration

```text
fps                    24
working frames         124
repair frames          72
context per side       60
guide frames           22
decoded feather        8 cosine frames
sampler                res_multistep
scheduler              simple
steps                  20
denoise                1.0
seed                   fixed
first test canvas      864×480 or 960×544
```

Outputs use unique prefixes:

```text
cauce/temporal/W0_join/<case>_<seed>
cauce/temporal/W1_mask/<case>_<seed>
cauce/temporal/W2_guides/<case>_<seed>
cauce/temporal/W3_continuous/<case>_<seed>
cauce/temporal/W4_native/<case>_<seed>
```

The browser workflow JSON and API prompt JSON are separate artifacts. Build the
browser graph first against the exact live schemas, save it, then export and
validate the API representation. Never hand-assume socket order from a stale
workflow.
