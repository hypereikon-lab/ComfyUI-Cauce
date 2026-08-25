# CAUCE engineering and laboratory runbook

This file applies to the complete repository. Read it before changing code,
deploying the node pack, or operating the laboratory ComfyUI instance. User
instructions remain authoritative.

## 1. Mission

CAUCE is a native ComfyUI operation pack. It owns reusable visual mathematics,
thin ComfyUI bindings, explicit H3 latent adapters, and bounded persistence.

The stable surface contains:

- phase-aware H3 continuation;
- exact decoded-range acceptance;
- localized temporal inpainting and duration-preserving splice mathematics;
- generic coordinate-map and vector-field operations;
- H3 audiovisual-latent save/load.

Six nodes live under `CAUCE/Research`. They are executable hypotheses, not
production presets:

- native-latent bidirectional seam preparation;
- direct H3 latent warp;
- motion-correlated H3 noise;
- sigma-conditioned latent transport;
- one-shot H3 visual clean-estimate injection during deterministic Euler flow
  sampling.

CAUCE does not own:

- a second UI or dashboard;
- production scheduling or editorial state;
- remote administration, authentication, or browser automation;
- model installation or runtime upgrades;
- semantic descriptions of images;
- generative sound, training, LoRAs, acceleration, or streaming.

The production soundtrack is fixed and remains outside H3 conditioning. H3's
packed structural-audio stream is still required by the model and must be
copied, frozen, or persisted correctly.

## 2. Sources of truth

Use this order:

1. `git status --short`, current branch, and recent commits;
2. this file;
3. `README.md`;
4. `docs/INDEX.md`;
5. `docs/ARCHITECTURE.md` and `docs/SYSTEM_BOUNDARIES.md`;
6. `docs/NODE_CATALOG.md` and `docs/CAPABILITY_MAP.md`;
7. the relevant mathematics or validation document;
8. code and tests;
9. the live ComfyUI runtime for live claims.

Do not reconstruct state from conversational memory alone. If the user points
to a Codex JSONL, search it narrowly for exact tool calls, commit hashes,
workflow signatures, Manager routes, and outputs. Never print authentication
cookies, Cloudflare credentials, or unrelated conversation content.

Prefer an official node plus graph composition over a CAUCE wrapper that only
renames an upstream operation or supplies defaults. Add a custom node only when
it owns mathematics, model translation, a safety boundary, or a reproducibility
guarantee.

## 3. Repository practice

Discover the root with `git rev-parse --show-toplevel`. Preserve dirty-worktree
changes. Do not use destructive checkout, reset, broad cleanup, or force-push.

The node registry is assembled only from:

```text
cauce_nodes/continuity.py
cauce_nodes/seams.py
cauce_nodes/motion.py
cauce_nodes/persistence.py
cauce_nodes/research.py
```

Stable nodes must not import project state or remote-runtime concerns. Research
nodes must use `CATEGORY = "CAUCE/Research"` and state their experimental status
in `DESCRIPTION`.

CAUCE intentionally ships no ComfyUI workflow JSON. New graphs are designed
after their operation contracts are understood. If reproducible graphs are
added later, treat browser-format JSON and API prompt JSON as different
artifacts and test them against live `/object_info` schemas.

Local verification:

```bash
python3 -m compileall -q cauce cauce_nodes
python3 -m unittest discover -s tests -v
git diff --check
```

Developer Python may lack NumPy or PyTorch. Do not install or upgrade global
GPU packages to satisfy local tests. Pure-Python tests must pass; tensor tests
may be run with a suitable isolated runtime or inside ComfyUI.

## 4. Architecture invariants

Dependency direction:

```text
ComfyUI graph
  -> cauce_nodes bindings
      -> cauce operations
          -> NumPy/PyTorch and official ComfyUI runtime hooks
```

Do not reverse this direction. Core operations must not know about browser tabs,
Cloudflare, Manager, queue routes, or a production project.

Node sockets should use local, explicit data:

- frames and frame counts;
- images, masks, latents, motion maps, and vector fields;
- sampler parameters;
- small serialized operation plans and reports.

Avoid global orchestration objects. If two nodes must share state, define the
smallest contract that represents the mathematical operation.

The package version lives in both `pyproject.toml` and `cauce.__version__`; keep
them equal.

## 5. H3 invariants

- Production video is 24 fps.
- Legal H3 visible-frame counts follow `17k + 5`.
- `124` visible frames represent about `5.1667` seconds and 37 visual latent
  tokens.
- The packed H3 state contains visual and structural-audio streams.
- Visual operations must not silently drop, regenerate, or spatially transform
  the structural-audio stream.
- Independent H3 latents do not automatically share causal phase.
- Use `cauce/timebase.py` for H3 frame/token arithmetic.

### Continuation

Continuation copies a phase-aligned visual tail into a target latent, sets that
context to preserved in the visual denoise mask, and freezes structural audio.
Official conditioning nodes remain outside CAUCE.

The parent latent must end at a visible-frame boundary on the H3 grid. Decoded
acceptance uses explicit `start_frame` and `frame_count` values.

### Temporal inpainting

The characterized operation is:

```text
two decoded 24 fps clips
-> tail/head working batch
-> encoded source video latent
-> binary or explicit continuous H3 per-row denoise support
-> official guide nodes wired outside CAUCE
-> sample masked interval
-> decode
-> decoded opacity feather
-> duration-preserving splice
```

For the measured 124-frame configuration:

```text
repair interval: [26, 98) = 72 frames
incoming guide:  [4, 26)  = 22 frames
outgoing guide:  [98, 120) = 22 frames
```

A soft decoded opacity feather is not a continuous denoise mask. Binary support
remains the default production control. An explicitly connected continuous
field may assign fractional denoise strength per H3 visual row; its temporal
values must be compiled on the H3 token grid. Do not conflate either sampling
mode with decoded compositing.

### Motion maps

Motion maps are inverse pullbacks. At output coordinate `x`, the map stores the
source coordinate to sample. Composition therefore follows function
composition, and images should normally be sampled once after maps are
composed.

Use normalized PyTorch `align_corners=False` coordinates. Preserve validity and
disocclusion fields throughout composition and resizing.

## 6. Research discipline

Research nodes stay experimental until they pass matched controls and an
operation-specific quality gate.

Required sequence:

1. official baseline;
2. CAUCE path with identity or zero strength;
3. prove identity or explain the residual;
4. activate one small intervention;
5. inspect decode integrity;
6. measure the requested effect;
7. increase magnitude only after the previous gate passes.

Current safe starting points:

```text
warped-noise temporal correlation: 0.05
motion-map envelope:               0.15
sigma transport strength:          0.10
sigma padding:                      border
```

A clean decode proves tensor compatibility, not motion obedience. Pixel
difference proves influence, not direction. Use optical flow, registration,
endpoint drift, or another measurement tied to the requested field.

## 7. Laboratory topology

The usual laboratory origin is:

```text
https://comfy.hypereikon.online/
```

Physical envelope:

```text
Windows portable ComfyUI
RTX 5090, 32 GB VRAM
64 GB RAM
Cloudflare Tunnel -> http://localhost:8188
Cloudflare Access in front of the hostname
```

The tunnel exposes the ComfyUI HTTP origin. It is not remote desktop, SSH,
PowerShell, CMD, or an arbitrary filesystem shell.

The hostname works only while the tower, network, tunnel service, and ComfyUI
process are all healthy. Manager can perform only the HTTP operations it
implements.

Use the user's authenticated in-app browser session for live operations. Do not
extract cookies or place credentials in shell commands, code, graph JSON, or
documentation.

## 8. Targeted deployment

Before deployment:

- user authorization covers the update;
- intended changes are committed and pushed to the repository branch followed
  by the installed clone; Manager installations normally follow the repository
  default branch, so verify that branch contains the intended commit;
- the Comfy queue is idle;
- the authenticated browser tab is on the laboratory origin;
- no unrelated core/runtime update is required.

Targeted Manager sequence:

```text
POST /manager/queue/reset
POST /manager/queue/update
POST /manager/queue/start
GET  /manager/queue/status
GET  /customnode/installed
```

Payload:

```json
{
  "id": "ComfyUI-Cauce",
  "ui_id": "ComfyUI-Cauce",
  "version": "unknown",
  "files": ["https://github.com/hypereikon-lab/ComfyUI-Cauce"]
}
```

Poll until Manager is idle and verify the reported CAUCE commit. Python changes
require:

```text
POST /manager/reboot
```

An immediate Cloudflare 502 is expected while ComfyUI restarts. Wait, reload,
then verify:

```text
GET /customnode/installed
GET /object_info/CauceBuildSeamWindow
GET /queue
```

Do not update CUDA, PyTorch, drivers, models, ComfyUI core, or unrelated custom
nodes as part of a CAUCE deployment. Do not reboot the physical tower.

## 9. Browser and workflow-tab hygiene

Distinguish:

1. browser page tab;
2. Comfy workflow tab inside the frontend;
3. automation handle controlling one browser page.

One browser page can contain many workflow tabs. Before live work, record:

```text
label | owner | purpose | identifying signature | output prefix | state
```

Every pre-existing or unidentified workflow is user-owned. Close only workflows
created by the current agent or whose provenance is exact.

Use at most one active experiment and one comparison graph. Pasting JSON can
open another workflow tab instead of replacing the current canvas; reconcile
the ledger immediately.

Before Run verify:

- active workflow label and unique signature;
- source filenames;
- model, sampler, scheduler, steps, seed, and denoise;
- output prefix;
- zero validation errors;
- idle queue.

After completion:

- enumerate exact outputs through `/history`;
- preserve reproducible graph state when required;
- close only agent-owned temporary tabs;
- leave no blank staging graph or stale experiment active.

The Assets sidebar is an index, not filesystem authority. Confirm outputs with
`/history` and authenticated `/view` requests.

## 10. Live experiment protocol

Change one independent variable at a time. A matched comparison holds fixed:

- source media and ordering;
- prompt;
- seed and noise source;
- model and quantization;
- resolution and frame count;
- sampler, scheduler, steps, and denoise;
- decode and save path.

Use these states:

- `graph validated`;
- `executes`;
- `executes but rejected`;
- `verified`;
- `blocked`.

Never call an inference successful only because the queue completed. For
temporal edits inspect the target interval, both patch edges, unchanged regions,
frame count, fps, and duration.

For long jobs, poll every 20–40 seconds, communicate at least once per minute,
and do not submit a second job because an aggregate percentage appears stalled.

## 11. Recovery

| Observation | Interpretation | Action |
| --- | --- | --- |
| brief 502 after Manager reboot | ComfyUI process is restarting | wait and reload |
| origin remains 502 | tunnel may be alive while ComfyUI is down | wait once, then request operator start |
| hostname remains unreachable | tower, network, or tunnel may be down | request smallest physical check |
| node absent from `/object_info` | import or dependency failure | inspect logs; do not mutate GPU stack blindly |
| Manager reports old commit | update did not reach installed clone | verify remote ref and rerun targeted update |
| queue unexpectedly busy | active or stalled GPU job | inspect queue/history before restart |

Retry only the narrow safe operation. Report the precise failed layer.

## 12. Finish checklist

- Code, registry, documentation, and tests agree.
- Pure-Python tests pass; skipped tensor dependencies are reported.
- `git diff --check` passes.
- Package versions match.
- No removed or unregistered node is referenced.
- No secret or authentication state entered the repository.
- For deployment, installed commit and representative `/object_info` agree.
- For live inference, history and visual/measurement gates were inspected.
- Agent-owned workflow tabs were cleaned up without touching user-owned tabs.
- No unrelated runtime component changed.
