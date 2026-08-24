# Capability map

This map identifies who owns each operation. CAUCE code is justified only when
the operation is not already expressed faithfully by an official node.

| Capability | Owner | CAUCE role |
| --- | --- | --- |
| H3 model loading | official ComfyUI | none |
| first/last-frame conditioning | official H3 node | none |
| ordered image/video references | official H3 node | none |
| guide application | official `MiniMaxH3AddGuide` | seam plan reports exact ranges |
| sampling and scheduling | official ComfyUI | none for stable surface |
| VAE encode/decode | official ComfyUI | validate compatible H3 geometry |
| phase-aware continuation | CAUCE adapter | copy protected visual tail, freeze structural audio |
| decoded frame acceptance | CAUCE operation | exact local slicing |
| temporal edit planning | CAUCE operation | legal H3 window, mask and splice geometry |
| per-token temporal mask | CAUCE H3 adapter | binary visual support, frozen structural audio |
| decoded seam splice | CAUCE operation | opacity feather + duration preservation |
| coordinate-map construction | CAUCE operation | reusable inverse pullbacks |
| motion-map composition | CAUCE operation | one final resample |
| image warp | CAUCE operation | grid sampling + validity |
| H3 AV latent persistence | CAUCE operation | bounded atomic safetensors |
| native-latent seam | CAUCE Research | experimental |
| H3 latent warp | CAUCE Research | experimental |
| warped H3 noise | CAUCE Research | experimental |
| sigma transport | CAUCE Research | experimental |
| remote access and deployment | laboratory runtime | documented separately |

## Arbitrary media

CAUCE never requires an inferred semantic record. An input image, video,
displacement field, depth map, or latent is treated according to its tensor and
socket contract only.

A video used as an H3 reference may carry filmed movement, a geometric grid,
simulation output, depth rendering, optical-flow visualization, or any other
motion signal. That interpretation belongs to graph composition and prompting,
not a CAUCE entity type.

## Extension rule

Before adding a node, answer:

1. Can official ComfyUI express it directly?
2. Can a graph composition express it without hidden semantics?
3. Does it own reusable mathematics or an H3 translation?
4. What matched test proves the operation?
5. Does it belong in stable or Research?

If the answer to the first or second question is yes and no unique contract is
added, do not create a wrapper.
