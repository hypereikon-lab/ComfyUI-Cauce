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

## Promotion rule

No Research operation is promoted by clean decode alone. Promotion requires a
matched baseline, identity control, measured intended effect, repeated visual
success, and a bounded resource envelope.
