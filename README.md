# CAUCE

**Media, time, and continuity for audiovisual generation in ComfyUI.**

CAUCE is a native ComfyUI custom-node pack developed by Hypereikon. It treats
images, video, audio, masks, prompts, and latents as opaque media placed on one
exact clock, then compiles that structure into current MiniMax H3 workflows.

CAUCE does not describe subjects, infer actions, or impose a shot ontology. The
model interprets media; CAUCE controls timing, topology, conditioning,
preservation, continuation, decoding, and provenance.

```text
media
  -> rational timeline
  -> generation windows
  -> native H3 conditioning
  -> AV masks / continuation
  -> accepted ranges / decode domains
  -> versioned artifacts
```

## Current status

The greenfield core and 39 ComfyUI nodes are implemented. Seven visual workflows,
two API templates, bounded demo assets, and a restartable two-window project are
included. Plate export, FL2VA first/last, Ref2VA landscape, and endpoint-guided
continuation have executed successfully on the live lab RTX 5090. Mask-only
continuation also executes, but its first visual seam failed the quality gate;
the shipped hybrid workflow passed that visual comparison. Its audio seam still
requires a listening gate before production promotion. Confluence seam repair
has completed end-to-end with synthetic inputs; heterogeneous real gesture
pairs remain the qualitative promotion gate.

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
- Dense H3 is the production baseline; research accelerators remain optional.

## Requirements

- A current ComfyUI build containing:
  - `MiniMaxH3ImageToVideo`
  - `MiniMaxH3ReferenceToVideo`
- Optional for `CAUCE · H3 Timed Guide`:
  - `MiniMaxH3AddGuide`
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
- **CAUCE / Audio** — sample-exact slicing, placement, mixing, and final
  authoritative audio.
- **CAUCE / Continuity** — phase-safe latent parents, masked AV continuation,
  and exact decoded AV acceptance.
- **CAUCE / Seams** — decoded-domain confluence windows, central H3 video
  inpainting, and duration-preserving local replacement.
- **CAUCE / Artifacts** — receipts and atomic nested AV latent save/load.
- **CAUCE / Runtime** — bounded 5090 profiles and read-only preflight.

See [Node catalog](docs/NODE_CATALOG.md),
[architecture](docs/ARCHITECTURE.md),
[H3 wiring](docs/H3_WIRING.md), and
[workflow guide](docs/WORKFLOWS.md),
[validation matrix](docs/VALIDATION.md), and
[laboratory results](docs/LAB_RESULTS.md). The current upstream and community
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
research packs are not copied into this repository; future integration with
them must remain an optional external dependency or be an independent
paper-derived implementation.
