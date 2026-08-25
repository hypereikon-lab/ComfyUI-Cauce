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

## W5 — H3 flow latent-injection ablation

W5 evaluates direct visual-latent intervention without changing H3 references,
guides, prompt, scheduler, target geometry, or structural audio.

```text
same-geometry guide video ─ H3 VAE encode ─ guide LATENT ─┐
official BasicScheduler ───────────────────────────────────┼─ CauceH3FlowLatentInjectionSampler
official KSamplerSelect(euler) ────────────────────────────┘

official H3 target + conditioning ─ SamplerCustomAdvanced ─ decode ─ save
```

Run three matched cases:

```text
W5-A  official Euler sampler, no CAUCE sampler adapter
W5-B  CAUCE adapter, strength = 0.00
W5-C  CAUCE adapter, strength = 0.05, then 0.10 only if W5-B is exact
```

Start with `flow_progress = 0.45`, full mask, and `mask_projection = mean`.
The percentage targets actual clean weight `1-sigma_next`, not a normalized
step index; connect the same `SIGMAS` to both the CAUCE adapter and
`SamplerCustomAdvanced`.
W5-A and W5-B must match exactly. W5-C must record the direction and magnitude
of its visual effect, not merely that its output differs. Keep at least one H3
model evaluation after the injection; the node enforces this structurally.

After the full-mask ablation, connect one standard animated Comfy mask to test
localized injection. The mask is projected to H3's causal visual-token spans;
fractional values remain fractional. The guide latent must have the exact target
`[B,C,T,H,W]` geometry. This experiment supports deterministic Euler only.

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
cauce/research/W5_flow_injection/<case>_<seed>
```

The browser workflow JSON and API prompt JSON are separate artifacts. Build the
browser graph first against the exact live schemas, save it, then export and
validate the API representation. Never hand-assume socket order from a stale
workflow.
