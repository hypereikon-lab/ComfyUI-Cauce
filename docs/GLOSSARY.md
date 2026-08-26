# Working vocabulary

Use terms that identify observable data or a documented operation.

| Term | Meaning |
| --- | --- |
| source frame | one input image at a defined frame index |
| anchor frame | a source frame intentionally placed at a generation boundary |
| frame batch | an ordered `IMAGE` tensor batch |
| source clip | decoded source frames plus an explicit frame rate |
| reference image / clip | media supplied to a documented H3 reference-conditioning path |
| guide frame / clip | media supplied through an official temporal guide node |
| target window | complete frame interval requested from H3 |
| context range | source or guide frames preserved as conditioning context |
| generated range | frames produced by the model inside a target window |
| accepted range | half-open decoded range retained after generation |
| frame range | half-open interval `[start, end)` |
| timeslot | editorial interval expressed as exact frame bounds at a known frame rate |
| native AV latent | H3's packed visual and structural-audio latent streams |
| native AV span | synchronized video/audio token subrange retaining absolute frame and clock bounds |
| denoise mask | nested continuous per-token video/audio strengths where `1` generates and `0` preserves |
| completion interval | explicit prefix, interior, or replacement range selected for generation inside a complete native target lattice |
| operation contract | typed reusable graph function with explicit node ownership and evidence state |
| operation invocation | project data binding one operation version to concrete media and parameters |
| UI graph | ComfyUI graph including layout and widget state |
| API graph | server-executable prompt graph produced from a UI graph |
| run receipt | immutable record of inputs, graph, runtime, queue result, and artifacts |

`Continuation` is reserved for generation that carries native model state or a
documented native overlap into a later generation. `Inpainting` is reserved for
an operation with source media, an explicit mask, and preserved regions.

Implementation names must identify the actual conditioning, latent, coordinate
map, or frame-range operation. An editorial description does not define a
technical mechanism.
