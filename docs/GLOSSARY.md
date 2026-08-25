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
| workflow spec | versioned, declarative intent independent of browser state |
| UI graph | ComfyUI graph including layout and widget state |
| API graph | server-executable prompt graph produced from a UI graph |
| run receipt | immutable record of inputs, graph, runtime, queue result, and artifacts |

`Continuation` is reserved for generation that carries native model state or a
documented native overlap into a later generation. `Inpainting` is reserved for
an operation with source media, an explicit mask, and preserved regions.

Do not use names such as *plate*, *confluence*, *seam fix*, or *continuity
engine* as technical mechanisms. Editorial prose may say that a generated
interval connects two clips, but implementation names must state the actual
conditioning and frame-range operation.
