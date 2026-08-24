# Bidirectional temporal inpainting

This document defines CAUCE's canonical operation for replacing a bounded
interval around a cut while preserving the total duration of both source
clips. It is a graph contract, not a model wrapper or an editorial object.

## 1. Operation

Let the decoded 24 fps sources be

```text
L = {L_0, ..., L_(nL-1)}
R = {R_0, ..., R_(nR-1)}
```

with the same height, width, and channel layout. Select `C` source frames per
side and a legal H3 working length `N = 17k + 5`. The working video is

```text
X = [left guard, L_(nL-C):nL, R_0:C, right guard]
```

where symmetric endpoint guards exist only when `2C < N`. The cut is at
`q = guard + C`.

Three intervals are deliberately independent:

```text
S = [s0,s1)   sampling support: rows allowed to denoise
A = [a0,a1)   accepted repair: decoded frames kept from the proposal
G_L, G_R      optional incoming/outgoing guide clips
```

The canonical decoded-source workflow currently uses `S = A`. A research
variant may use `A ⊂ S` as temporal overscan, but it must not silently change
the final replacement duration.

## 2. Measured 124-frame geometry

With 2.5 seconds of source context per side, a three-second repair, and
22-frame guides:

```text
N                         124 frames
C                          60 frames per source
guard                       2 frames per side
cut                         62
S = A                      [26,98) = 72 frames
G_L                         [4,26) = 22 frames
G_R                         [98,120) = 22 frames
```

All ranges are half-open. The seam plan is the source of truth; downstream
nodes must not recompute these values from approximate seconds.

## 3. Latent construction

Encode `X` once with the H3 video VAE:

```text
Z_X = E_video(X) ∈ R^[B,24,Tv,H/16,W/16]
```

Create an official target H3 audiovisual latent with matching geometry, then
replace only its visual stream with `Z_X`. The structural-audio carrier remains
present because H3 samples a packed audiovisual state, but its denoise mask is
zero and its decoded audio is discarded. The production soundtrack is never
sent through this operation.

Define visible sampling support

```text
m_f = 1  when f ∈ S
      0  otherwise.
```

CAUCE projects this field onto H3 temporal latent spans `(1,4,4,4,4)` and emits
a nested AV noise mask. The canonical temporal operation is binary. Spatial
masks may be attached independently, but ComfyUI pools them to the H3 DiT's
`2×2` latent-patch row grid.

Current ComfyUI core implements the row timestep as

```text
t_i(σ) = min(1 - m_i σ, t_pin)
```

where a generated row (`m_i = 1`) follows the sampling schedule and a preserved
row (`m_i = 0`) remains near H3's visual conditioning timestep. ComfyUI also
reinjects the clean latent in preserved regions during sampling. This is the
mechanism that makes the operation conditional inpainting rather than ordinary
video-to-video denoising.

## 4. Conditioning is separate from preservation

The source latent and the guide clips have different roles:

- the masked main latent defines exactly which rows are known and unknown;
- `MiniMaxH3AddGuide` adds encoded clips as extra conditioning rows at declared
  frame indices;
- the prompt supplies semantic and motion intent.

Because these channels can be redundant or conflicting, every material test
must compare:

```text
W0  hard concatenation, no generation
W1  masked H3 temporal inpainting, no AddGuide clips
W2  the same mask plus incoming and outgoing 22-frame guides
```

`W2` is the current production candidate, not an axiom. A guide clip must use a
legal `17k+5` length; 22 frames is the shortest useful motion-bearing guide in
this contract.

## 5. Prompt contract

Prompting stays an explicit workflow input. CAUCE does not describe or classify
the images. The first general production prompt is:

```text
A single continuous shot. Preserve the scene, subjects, composition, camera
path, lighting, scale, motion direction and motion speed. Repair only the
transition across the edit. Motion flows continuously through the regenerated
interval without a stop, reset, new shot or cut. Introduce no new objects and
do not change the visual style.
```

Also run an empty/minimal-prompt control. If the detailed prompt changes
content that the mask and guides already determine, reduce it rather than
adding more semantic instructions.

## 6. Sampling contract

Start from the official quality baseline:

```text
model       FL2VA INT8/pruned INT8 when required by the 32 GB envelope
sampler     res_multistep
scheduler   simple
steps       20
denoise     1.0
seed        fixed per comparison set
length      124
fps         24
```

Do not begin with an alternate sampler, a turbo LoRA, low denoise, or several
simultaneous interventions. Native H3 row masking in ComfyUI core is the
canonical sampling path for this operation.

Run structural tests at 864×480 or 960×544. Promote a setting to 1344×768 only
after its geometry and qualitative effect pass.

## 7. Decode and duration-preserving splice

Decode the complete proposal `Y`. Keep only `Y_A`, and combine it with the
original frames occupying the same interval. A decoded opacity field `α` may
use cosine, smoothstep, or linear ramps:

```text
P_f = (1 - α_f) X_f + α_f Y_f,  f ∈ A.
```

The opacity feather is output compositing. It does not alter H3's sampling mask
and must never be described as soft latent denoising. Outside `A`, the output
must be byte/tensor-identical to the original decoded input batch. The output
frame count is exactly `nL + nR`.

## 8. Validation

### Structural invariants

- both sources are 24 fps and share canvas geometry;
- `N` lies on `17k+5`;
- `S` begins and ends on H3 temporal-token boundaries;
- `G_L` and `G_R` fit completely and do not overlap `S`;
- the structural-audio mask is zero;
- no frame outside `A` changes;
- total duration is preserved.

### Comparative measurements

For every seed, retain W0/W1/W2 and inspect:

- optical-flow direction and magnitude immediately before and after both patch
  boundaries;
- acceleration or flow discontinuity at the boundaries;
- image/feature drift outside the accepted interval;
- endpoint fidelity inside the decoded feather;
- runtime and peak VRAM;
- new cuts, motion resets, hallucinated objects, or composition changes.

A smooth crossfade is not proof of motion continuity. The accepted result must
carry compatible velocity through both boundaries.

## 9. Limits and implications

The H3 video VAE is causal and compresses unequal visible-frame spans into
temporal latents. Exact pixel-frame control therefore requires token-aligned
sampling ranges and a separate decoded acceptance range. Copying arbitrary
latent rows between independently encoded runs is not automatically safe;
native-latent seams remain Research.

The current operation repairs one bounded interval. Long-form construction is
built by applying the same local contract repeatedly with explicit source,
seed, prompt, parameters, and accepted-output records.
