# Resultados medidos en el laboratorio

Fecha de esta matriz: 2026-08-22. Instancia remota: ComfyUI `0.33.1`, frontend
`1.48.7`, Windows portable, Python `3.12.10`, PyTorch `2.13.0+cu130`, RTX 5090
de 32 GB y 64 GB de RAM. CAUCE no modificó CUDA, PyTorch, ComfyUI ni modelos.

| Recorrido | Estado | Render | Tiempo | Artifact principal |
|---|---|---:|---:|---|
| Plate sketch | verificado | PNG 1344×768 | CPU | `cauce/plates/forest_threshold_001_...png` |
| FL2VA first/last | verificado | 124 f, 640×640, 5,167 s | 97,6 s | `cauce/sequence/forest_window_001_00001_.mp4` |
| Ref2VA motion, square | ejecuta; aspect no aprobado | 124 f, 448×448, 5,167 s | 71,1 s | `cauce/demos/ref2va_motion_00001_.mp4` |
| Ref2VA motion, landscape | verificado | 124 f, 576×320, 5,167 s | 72,6 s | `cauce/demos/ref2va_motion_landscape_00001_.mp4` |
| Continuation `mask_only` | ejecuta; seam rechazado | 85 f aceptados, 640×640 | 93,4 s | `cauce/sequence/forest_window_002_00001_.mp4` |
| Continuation `mask_plus_guide` | incompatible con core 0.33.1 | — | falla antes de output | error de geometría en sampler |
| Continuation tail + endpoint | seam visual verificado; escucha pendiente | 85 f aceptados, 640×640, 3,542 s | 104,1 s | `cauce/sequence/forest_window_002_hybrid_00001_.mp4` |
| Timed Guide | bloqueado por capability | — | — | falta `MiniMaxH3AddGuide` oficial |
| Bridge | graph validado | — | — | pendiente GPU |
| Confluence seam repair, sintético | runtime verificado; gesto real pendiente | 124 f working → 120 f join / 48 f patch | 42,8 s a 4 steps | previews temporales, prompt `5ea6dc99-...` |

## Lectura de los resultados

El baseline FL2VA produjo video y audio, latent AV persistente y receipt. El
Ref2VA también completó, pero referencias landscape sobre un output cuadrado
indujeron letterboxing visible. El segundo run `576×320` eliminó ese problema,
ocupó 72,6 s y produjo un MP4 de 1.361.003 bytes; ese profile queda verificado
para el envelope demo de 124 frames.

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
AV y fase causal. El run híbrido conservó la composición del río, el árbol
central, la perspectiva y la dirección de avance en el primer frame aceptado de
B, por lo que pasa este primer seam gate visual. Receipt:
`5d56dccb564f350687e10a5d8d760c219fe1a4ca30983d93910b0689db6a69e4`.

El borde de audio fue decodificado a 44,1 kHz estéreo. En una ventana de 250 ms,
el RMS cambió de `0,0119` a `0,0246`, el salto de muestra fue `0,0102` y la
correlación normalizada fue `-0,0144`. No aparece un impulso extremo, pero esas
métricas no prueban continuidad perceptual; queda una escucha crítica pendiente.

Confluence se ejecutó además con dos campos de color de 60 frames generados en
memoria, sin subir assets ni escribir un MP4 permanente. El build produjo el
dominio H3 esperado de 124 frames; `VAEEncode`, la inyección del video latent,
la máscara central, el sampler a 4 steps, el decode y el splice terminaron en
42,8 s. El preview final reportó 120 frames y el preview del parche 48, por lo
que la duración y el rango de reemplazo pasan el gate estructural. La calidad
temporal no se promueve hasta repetir con pares heterogéneos de gestos reales.

Receipt Ref2VA landscape:
`357751bc96407eda0532dcc7b7a2b459c25838106a7a1a654f56daff219d6bc1`.

## Reglas de promoción

- `verified`: ejecuta, produce artifacts completos y pasa inspección relevante.
- `executes`: el runtime termina, pero existe un problema de aspecto o calidad.
- `graph validated`: sockets y clases resuelven; todavía no hubo sample.
- `blocked`: falta una capability explícita; no se modifica el entorno para
  ocultarlo.
