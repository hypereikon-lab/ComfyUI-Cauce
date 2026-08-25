# CAUCE

CAUCE is a small native ComfyUI node pack for deterministic media preparation,
H3 two-sided guide-window assembly, reusable motion-reference construction, and
packed H3 audiovisual-latent persistence.

It deliberately leaves MiniMax H3 conditioning and sampling to the official
ComfyUI nodes. CAUCE does not wrap a sampler, modify H3 latents, or claim that a
completed queue item is a successful visual result.

## Node surface

The package registers 15 nodes:

- 10 coordinate-map and image-warp nodes under `CAUCE/Motion Maps`;
- 2 two-sided guide-window nodes under `CAUCE/Native H3`;
- 2 packed audiovisual-latent save/load nodes under `CAUCE/Persistence`;
- 1 exact decoded-range node under `CAUCE/Assembly`.

## H3 two-sided guide window

`CaucePrepareH3TwoSidedGuideWindow` extracts the final guide clip from source A and the
initial guide clip from source B. The graph supplies them to two official
`MiniMaxH3AddGuide` nodes on a fresh H3 target. After normal official sampling
and decoding, `CauceAssembleH3TwoSidedGuideWindow` discards both guide intervals, accepts
only the generated center, and assembles:

```text
complete A + generated center + complete B
```

The default geometry is:

```text
target                         124 frames at 24 fps
A tail guide                    22 frames at frame_idx 0
B head guide                    22 frames at frame_idx 102
accepted generated center       80 frames [22, 102)
```

This is a deterministic graph contract around an official conditioning
mechanism. Its visual quality remains an empirical result that must be checked
for every source pair.

## Install

Install the repository as a ComfyUI custom node and restart the ComfyUI Python
process. No additional Python package is declared by CAUCE.

```text
https://github.com/hypereikon-lab/ComfyUI-Cauce
```

## Documentation

- [Documentation index](docs/INDEX.md)
- [Architecture and boundaries](docs/ARCHITECTURE.md)
- [Native H3 workflow recipes](docs/NATIVE_H3_WORKFLOWS.md)
- [Node catalog](docs/NODE_CATALOG.md)
- [Motion-reference maps](docs/MOTION_MAPS.md)
- [Validation protocol](docs/VALIDATION.md)
- [Remote ComfyUI runtime](docs/REMOTE_COMFY_RUNTIME.md)

## Local verification

```bash
python3 -m compileall -q cauce cauce_nodes
python3 -m unittest discover -s tests -v
git diff --check
```
