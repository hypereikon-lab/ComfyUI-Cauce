# Technical language

Use names that describe the actual operation.

| Preferred term | Meaning |
| --- | --- |
| guide clip | decoded IMAGE batch supplied to `MiniMaxH3AddGuide` |
| native tail continuation | extension using a prior native H3 tail through supported guide conditioning |
| native AV completion | generation of an explicit prefix/interior/replacement interval while known native AV tokens are preserved by the official mask path |
| denoise interval | pixel-frame range mapped independently to continuous H3 video/audio token masks |
| video denoise mask | continuous spatial or spatiotemporal selection projected to the native H3 visual-token lattice; `1` generates and `0` preserves |
| local retake | bounded masked re-denoising of an existing source state; a workflow variant, not a special sampler |
| video outpaint | expansion of the native visual lattice followed by generation only in newly allocated regions |
| video refinement | a bounded-denoise second H3 pass retaining the source native state as sampler input |
| placed AV span | synchronized native context copied to an exact compatible target interval |
| AV window layout | absolute frame interval and its synchronized video/audio token boundaries |
| AV span | typed synchronized tensor subrange retaining absolute timeline bounds |
| coordinate pullback | target-to-source sampling map |
| structural-audio stream | packed H3 internal stream, distinct from the fixed production soundtrack |

Evidence labels:

```text
unit-validated
schema-validated
executes
visually accepted
rejected
blocked
```

Avoid metaphorical feature names when a precise operation name exists. Do not
use “works,” “fixed,” “seamless,” or “production-ready” without naming the gate
that supports the claim.
