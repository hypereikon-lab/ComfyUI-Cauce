# Laboratory results

This document records conclusions that still constrain current CAUCE
operations. A completed queue job is not automatically a verified result.

## Runtime

The laboratory instance has executed current MiniMax H3 inference on an RTX
5090 with 32 GB VRAM and 64 GB system RAM. The ComfyUI process is reached through
Cloudflare Access and Tunnel; Manager updates and ComfyUI-only restarts have
worked through the authenticated origin.

## Temporal inpainting

The strongest characterized path uses decoded source context, official H3
per-token denoise masks, bidirectional guide clips, and a decoded opacity
feather.

Measured 124-frame geometry:

```text
working domain: [0,124)
repair:         [26,98)
left guide:     [4,26)
right guide:    [98,120)
```

The repair interval is 72 frames, or exactly three seconds at 24 fps. The final
splice preserves the combined source duration and leaves regions outside the
replacement unchanged.

Important result: a binary latent denoise mask plus a soft decoded feather
performed better than treating temporal sampling strength as a broad soft
gradient. The model receives an unambiguous unknown interval; opacity blending
handles only the decoded patch edges.

## Native-latent seam

The direct native-latent path validates phase-aligned extraction, AV packing,
binary center masking, and clean decoding. Its perceptual seam quality has not
yet passed a production gate. It remains Research.

## Motion maps

Affine, projective, analytic, displacement, advection, depth-camera, modulation,
composition, and image-warp mathematics have deterministic unit coverage.

Direct H3 latent interventions remain experimental:

- a small sequential latent warp followed by a repair pass decoded coherently;
- strong warped-noise settings corrupted the result;
- weak warped-noise settings stayed coherent but did not prove reliable motion
  obedience.

Safe starting values are a `0.05` temporal correlation and approximately `0.15`
map modulation.

## Sigma transport

The zero-strength Euler integration matched the official output bit-for-bit.
Small active transport changed the output while keeping decode integrity, but
larger displacements rapidly introduced tearing and mosaic artifacts.

Current conclusion: sigma transport is a material experiment, not a dependable
camera-control mechanism.

## H3 flow latent injection — W5

The first W5 ablation ran against live CAUCE commit
`6c604413572cec8f7119a823eb15d108e50adb6a` on 2026-08-24.

Fixed configuration:

```text
model                minimax_h3_fl2va_pruned_fp8_scaled.safetensors
text encoder         qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
video VAE            minimax_h3_video_vae_fp16.safetensors
canvas               608 x 352
frames / fps          124 / 24
sampler / scheduler   Euler / simple
steps / seed          20 / 8242026
inject_percent        0.45
guide                 the source still repeated to 124 frames, then VAE encoded
```

A four-step structural smoke test proved the neutral control exactly. Native
Euler and the connected CAUCE adapter at `strength = 0` produced identical
safetensors files, identical visual tensors, and identical structural-audio
tensors. An active `0.05` branch diverged and decoded successfully.

The matched 20-step active branches produced:

| strength | visual relative L2 vs native | visual cosine | decoded SSIM | decoded PSNR |
| ---: | ---: | ---: | ---: | ---: |
| `0.05` | `0.031264` | `0.999512` | `0.931254` | `34.528 dB` |
| `0.10` | `0.033631` | `0.999437` | `0.927821` | `34.074 dB` |

The intervention had a small directionally consistent effect for this static
guide. Adjacent-frame mean absolute difference fell from `0.029423` natively
to `0.029271` at `0.05` and `0.029051` at `0.10`. Mean structural correlation
to the first decoded frame rose from `0.564757` to `0.567665` and `0.570326`.
The operator therefore behaves here as a weak attraction toward the guide and
a slight damping of temporal change, not as an arbitrary motion controller.

A fourth 20-step branch used a left-side mask with a 96-pixel feather and
`strength = 0.10`. Relative to the full-mask intervention, the component of its
decoded perturbation aligned with the full intervention fell from `0.529` in
the unfeathered left core, to `0.454` through the feather, to `0.383` on the
unmasked right side. The mask therefore modulates the intended intervention,
but the later full-sequence H3 evaluation propagates consequences globally; it
is not a hard output-space isolation boundary. Native-versus-masked decode was
`SSIM 0.934894`, `PSNR 34.914 dB`.

Live jobs:

```text
four-step identity smoke   f8386193-57c2-4d01-8b5f-3c5a8dc02ccf
20-step full-mask matrix   ea18d1d8-c2c1-40a9-868c-f8e4d368875d
full-mask decode           bc1826f9-7532-41b3-bbc3-927a47638219
left-mask branch           f7441870-04be-4c75-9327-8aaf402c2dd8
left-mask decode           a287a36b-0d28-4cc0-8218-db79a7b74020
```

Conclusion: tensor geometry, exact identity, dose response, continuous-mask
projection, and clean decoding are established. W5 remains Research because a
static repeated-frame guide only tests structural attraction. The next causal
gate is a coordinate-warped same-geometry guide with a known displacement,
measured by registration or optical flow against both native Euler and the
zero-strength control.

## Promotion rule

No Research operation is promoted by clean decode alone. Promotion requires a
matched baseline, identity control, measured intended effect, repeated visual
success, and a bounded resource envelope.
