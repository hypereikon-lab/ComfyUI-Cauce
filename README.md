# CAUCE

CAUCE is a native ComfyUI operation pack for visual continuity, localized
temporal inpainting, motion-field construction, and H3 audiovisual-latent
persistence.

It provides operations, not a production application. CAUCE has no separate UI,
project model, timeline, workflow suite, remote client, model installer, or
semantic image ontology. Model loading and conditioning use official ComfyUI
nodes directly.

## Surface

The installed package registers 26 nodes:

- 20 stable nodes under `CAUCE/Continuity`, `CAUCE/Temporal Inpainting`,
  `CAUCE/Motion Maps`, and `CAUCE/Persistence`;
- 6 experimental nodes under `CAUCE/Research`.

Stable operations:

- phase-aware H3 continuation and exact decoded-range acceptance;
- duration-preserving temporal inpainting across a video cut;
- token-aligned continuous temporal denoise fields, composable with native
  animated masks;
- affine, projective, analytic, displacement, depth-camera, and advected motion
  maps;
- motion-map modulation, composition, and decoded image warping;
- atomic save/load of nested H3 visual and structural-audio latents.

Research operations:

- native-latent bidirectional seam preparation;
- direct H3 latent warping;
- motion-correlated H3 noise;
- sigma-conditioned latent transport;
- one-shot H3 visual clean-estimate injection during deterministic Euler flow
  sampling.

Research nodes execute real tensor paths but do not carry a production-quality
or motion-obedience guarantee.

## Principles

- Inputs remain opaque media. CAUCE does not infer subjects, actions, or shot
  descriptions.
- Motion maps are inverse pullbacks in normalized PyTorch
  `align_corners=False` coordinates.
- Temporal denoise strength and decoded opacity blending are distinct fields.
- Independent H3 latents are never treated as safely concatenable by default.
- H3 structural audio is preserved or frozen; it is not production audio.
- Official ComfyUI/H3 nodes are used whenever they already own the operation.

## Installation

Install with ComfyUI Manager from:

```text
https://github.com/hypereikon-lab/ComfyUI-Cauce
```

Restart ComfyUI after installation or update. CAUCE relies on the NumPy,
PyTorch, and safetensors runtime already supplied by ComfyUI.

## Local verification

```bash
python3 -m compileall -q cauce cauce_nodes
python3 -m unittest discover -s tests -v
git diff --check
```

Tensor tests are skipped when the developer Python lacks NumPy or PyTorch; the
ComfyUI runtime supplies those dependencies.

## Documentation

- [Documentation index](docs/INDEX.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Node catalog](docs/NODE_CATALOG.md)
- [Operations guide](docs/OPERATIONS_GUIDE.md)
- [Bidirectional temporal inpainting](docs/TEMPORAL_INPAINTING.md)
- [Workflow contracts](docs/WORKFLOW_CONTRACTS.md)
- [Motion-map mathematics](docs/MOTION_MAPS.md)
- [H3 flow latent injection](docs/H3_FLOW_LATENT_INJECTION.md)
- [Validation](docs/VALIDATION.md)
- [Remote ComfyUI runtime](docs/REMOTE_COMFY_RUNTIME.md)

## License

MIT. See [LICENSE](LICENSE).
