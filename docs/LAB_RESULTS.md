# Resultados medidos en el laboratorio

Fecha de esta matriz: 2026-08-22. Instancia remota: ComfyUI `0.33.1`, frontend
`1.48.7`, Windows portable, Python `3.12.10`, PyTorch `2.13.0+cu130`, RTX 5090
de 32 GB y 64 GB de RAM. CAUCE no modificó CUDA, PyTorch, ComfyUI ni modelos.

| Recorrido | Estado | Render | Tiempo | Artifact principal |
|---|---|---:|---:|---|
| Plate sketch | verificado | PNG 1344×768 | CPU | `cauce/plates/forest_threshold_001_...png` |
| FL2VA first/last | verificado | 124 f, 640×640, 5,167 s | 97,6 s | `cauce/sequence/forest_window_001_00001_.mp4` |
| Ref2VA motion, square | ejecuta; aspect no aprobado | 124 f, 448×448, 5,167 s | 71,1 s | `cauce/demos/ref2va_motion_00001_.mp4` |
| Continuation `mask_only` | ejecuta; seam rechazado | 85 f aceptados, 640×640 | 93,4 s | `cauce/sequence/forest_window_002_00001_.mp4` |
| Continuation `mask_plus_guide` | incompatible con core 0.33.1 | — | falla antes de output | error de geometría en sampler |
| Timed Guide | bloqueado por capability | — | — | falta `MiniMaxH3AddGuide` oficial |
| Bridge | graph validado | — | — | pendiente GPU |

## Lectura de los resultados

El baseline FL2VA produjo video y audio, latent AV persistente y receipt. El
Ref2VA también completó, pero referencias landscape sobre un output cuadrado
indujeron letterboxing visible; por eso el ejemplo cambia a un profile
`576×320` que debe medirse antes de promoverlo.

La primera continuación demostró que copiar y preservar 39 frames AV es válido
en términos de tensores y sampling, pero no basta como condición futura: el
primer frame aceptado de B cambió fuertemente de paisaje respecto del último de
A. Ése es un fail de calidad, aunque el job figure `success`.

El intento de añadir toda la cola latente como guide reveló una diferencia de
capabilities. ComfyUI 0.33.1 sólo empaqueta keyframes visuales first/last de un
frame; el core upstream posterior incorpora clip/audio guides. CAUCE ahora
detecta esa ausencia y falla con un mensaje dirigido, sin pedir ni efectuar una
actualización de la torre.

La estrategia compatible a validar usa dos señales del mismo parent:

```text
parent A
  ├─ tail AV de 39 frames → target B + máscara preserve
  └─ último frame decodificado → FL2VA first_frame de B
```

El segundo camino fortalece el borde perceptual; el primero conserva historia
AV y fase causal. Sólo una inspección del nuevo output puede determinar si la
combinación supera el seam gate.

## Reglas de promoción

- `verified`: ejecuta, produce artifacts completos y pasa inspección relevante.
- `executes`: el runtime termina, pero existe un problema de aspecto o calidad.
- `graph validated`: sockets y clases resuelven; todavía no hubo sample.
- `blocked`: falta una capability explícita; no se modifica el entorno para
  ocultarlo.
