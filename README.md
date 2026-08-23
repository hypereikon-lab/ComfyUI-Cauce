# CAUCE

**Media, time, and visual continuity against a fixed master clock in ComfyUI.**

CAUCE is a native ComfyUI custom-node pack developed by Hypereikon. It treats
images, video, masks, prompts, and latents as opaque media placed against one
exact clock, then compiles that structure into current MiniMax H3 workflows.
Production audio is an immutable master track: CAUCE may slice or mux it for
delivery, but visual generation never replaces it with H3-generated audio.

CAUCE does not describe subjects, infer actions, or impose a shot ontology. The
model interprets media; CAUCE controls timing, topology, conditioning,
preservation, continuation, decoding, and provenance.

```text
media
  -> rational timeline
  -> generation windows
  -> native H3 conditioning
  -> temporal denoise masks / continuation
  -> accepted ranges / decode domains
  -> versioned artifacts
```

## Current status

The greenfield core and 40 ComfyUI nodes are implemented. Six visual workflows,
two API templates, bounded demo assets, and a restartable two-window project are
included. Plate export, FL2VA first/last, Ref2VA landscape, and endpoint-guided
continuation have executed successfully on the live lab RTX 5090. Mask-only
continuation also executes, but its first visual seam failed the quality gate;
the shipped hybrid workflow passed that visual comparison. Localized temporal
inpainting is validated live on the updated official H3 runtime: it VAE-encodes
the existing video, regenerates only a binary token-aligned interval, conditions
on preserved guide clips immediately before and after it, and inserts the decoded
patch without changing total duration. The verified 3-second run used a
124-frame working window, 72 generated frames, 22-frame guide clips on both
sides, 20 sampling steps, and denoise 1.0 inside the mask only.

The previous Hypereikon H3 repository is not a dependency and no compatibility
layer is included.

## Principles

- ComfyUI-native nodes; no separate dashboard or timeline UI.
- Standard `IMAGE`, `VIDEO`, `AUDIO`, `MASK`, `LATENT`, and `CONDITIONING`
  sockets wherever media travels.
- Versioned `CAUCE_*` data only for plans, fields, windows, profiles, and
  receipts.
- Rational time and explicit frame/sample rounding.
- H3 integration through the current official ComfyUI nodes, not a copied fork.
- No automatic CUDA, PyTorch, driver, ComfyUI, or model installation.
- Retry-safe atomic latent and receipt persistence.
- Generated audio, training, LoRAs, acceleration, and streaming are outside the
  current production scope.

## Requirements

- A current ComfyUI build containing:
  - `MiniMaxH3ImageToVideo`
  - `MiniMaxH3ReferenceToVideo`
  - `MiniMaxH3AddGuide`
  - official MiniMax H3 per-token denoise-mask support (upstream PR #15375)
- The official H3 video/audio VAEs, text encoder, and selected diffusion model.
- The packages already shipped with a current ComfyUI portable runtime:
  PyTorch, Pillow, NumPy, torchaudio, and safetensors.

CAUCE introduces no pip dependency of its own.

## Install during development

Place or symlink this repository under `ComfyUI/custom_nodes/ComfyUI-Cauce`,
then restart ComfyUI once.

Manager URL:

```text
https://github.com/hypereikon-lab/ComfyUI-Cauce
```

The laboratory installation is tracked by ComfyUI Manager under this URL.

## Node groups

- **CAUCE / Timeline** — project, points, spans, exact H3 windows, timelines,
  and decode domains.
- **CAUCE / Media** — minimal opaque-batch operations such as exact frame
  selection.
- **CAUCE / Plates** — canvas, layers, masks, dome preview, point versions, and
  PNG/prompt handoff.
- **CAUCE / H3** — native FL2VA, ordered Ref2VA references, and absolute-time
  AddGuide.
- **CAUCE / Masks** — rational time fields and nested H3 video/audio masks.
- **CAUCE / Audio** — sample-exact handling of the fixed, authoritative master;
  never a source of generative audio.
- **CAUCE / Continuity** — phase-safe visual parents, masked continuation, and
  exact decoded acceptance; H3 audio rows stay frozen.
- **CAUCE / Temporal Inpainting** — localized H3 temporal denoise masks,
  bidirectional guide clips, decoded output opacity, and duration-preserving
  patch insertion.
- **CAUCE / Artifacts** — receipts and atomic nested AV latent save/load.
- **CAUCE / Runtime** — bounded 5090 profiles and read-only preflight.

See [Node catalog](docs/NODE_CATALOG.md),
[architecture](docs/ARCHITECTURE.md),
[H3 wiring](docs/H3_WIRING.md), and
[workflow guide](docs/WORKFLOWS.md),
[validation matrix](docs/VALIDATION.md), and
[laboratory results](docs/LAB_RESULTS.md). The canonical terminology and run
report format are defined in [technical language](docs/TECHNICAL_LANGUAGE.md).
The current upstream and community
comparison is recorded in [research notes](docs/RESEARCH.md).

## Tests

```bash
python -m unittest discover -s tests -v
```

## Sequence runner

CAUCE also ships a small non-UI runner for repeatable, restartable Comfy API
workflows:

```bash
python cauce_cli.py status examples/project.example.json
python cauce_cli.py resume examples/project.example.json --dry-run
python cauce_cli.py resume project.json
```

Run it on the laboratory machine against `http://127.0.0.1:8188`, or configure
a Cloudflare Access service token through the optional environment variables
documented in [Runner](docs/RUNNER.md). Browser login cookies are intentionally
not read or reused.

## License

CAUCE code is MIT licensed. Model weights retain their own licenses. GPL
research packs inspected during development are not copied into this
repository.
