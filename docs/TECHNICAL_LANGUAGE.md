# Technical language

Use names that describe the actual operation.

| Preferred term | Meaning |
| --- | --- |
| H3 two-sided guide window | fresh H3 target conditioned by two official temporal guides |
| guide clip | decoded IMAGE batch supplied to `MiniMaxH3AddGuide` |
| accepted generated range | decoded interval between the two guide spans |
| native tail continuation | extension using a prior native H3 tail through supported guide conditioning |
| AV window layout | absolute frame interval and its synchronized video/audio token boundaries |
| AV span | typed synchronized tensor subrange retaining absolute timeline bounds |
| motion-reference media | images/video geometrically prepared before official H3 conditioning |
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
