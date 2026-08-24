# Resultados medidos en el laboratorio

Fecha de esta matriz: 2026-08-23. La instancia remota comenzó la jornada en
ComfyUI `0.33.1`, frontend `1.48.7`, Windows portable, Python `3.12.10`,
PyTorch `2.13.0+cu130`, RTX 5090 de 32 GB y 64 GB de RAM. El core de ComfyUI se
actualizó posteriormente para disponer de las máscaras H3 oficiales por token y
`MiniMaxH3AddGuide`; CAUCE no modificó CUDA, PyTorch, drivers ni modelos.

| Recorrido | Estado | Render | Tiempo | Artifact principal |
|---|---|---:|---:|---|
| Plate sketch | verificado | PNG 1344×768 | CPU | `cauce/plates/forest_threshold_001_...png` |
| FL2VA first/last | verificado | 124 f, 640×640, 5,167 s | 97,6 s | `cauce/sequence/forest_window_001_00001_.mp4` |
| Ref2VA motion, square | ejecuta; aspect no aprobado | 124 f, 448×448, 5,167 s | 71,1 s | `cauce/demos/ref2va_motion_00001_.mp4` |
| Ref2VA motion, landscape | verificado | 124 f, 576×320, 5,167 s | 72,6 s | `cauce/demos/ref2va_motion_landscape_00001_.mp4` |
| Continuation `mask_only` | ejecuta; seam rechazado | 85 f aceptados, 640×640 | 93,4 s | `cauce/sequence/forest_window_002_00001_.mp4` |
| Continuation `mask_plus_guide` | incompatible con core 0.33.1 | — | falla antes de output | error de geometría en sampler |
| Continuation tail + endpoint | seam visual verificado | 85 f aceptados, 640×640, 3,542 s | 104,1 s | `cauce/sequence/forest_window_002_hybrid_00001_.mp4` |
| Timed Guide | bloqueado por capability | — | — | falta `MiniMaxH3AddGuide` oficial |
| Temporal inpainting, máscara estándar inicial | runtime sintético verificado; gesto real rechazado | 124 f working → 120 f join / 48 f patch | 42,8 s a 4 steps | previews temporales, prompt `5ea6dc99-...` |
| Temporal inpainting, LanPaint + campos continuos | rechazado por revisión de implementación | — | — | LanPaint umbraliza el denoise mask; la supuesta fuerza continua no llega al sampler |
| Temporal inpainting, overscan binario | runtime real; gesto rechazado | 124 f working / 72 f sampling / 48 f aceptación → 248 f join | 487,85 s | output histórico rechazado |
| Temporal inpainting, máscara H3 nativa + guías bidireccionales | verificado live | 124 f working / 72 f sampling y aceptación / guías 22+22 → 248 f join | 135,5 s | output histórico aprobado |
| Motion maps, composición vs dos resamples | verificado live | 124 f, 768×512 | 57,73 s | `cauce/motion_maps/70_composed_once_00001_.mp4` + `70_sequential_two_pass_00001_.mp4` |
| Depth-camera + RK4 advection preview | verificado live | 124 f, 768×512 | 22,12 s | `cauce/motion_maps/73_depth_advection_00001_.mp4` |
| H3 warped noise, stress `0,85`/`0,7` | ejecuta; decode rechazado | 124 f, 768×512 | 93,86 s | `cauce/motion_maps/71_h3_warped_noise_00001_.mp4` |
| H3 warped noise, seguro `0,05`/`0,15` | decode válido; control pendiente de métrica | 124 f, 768×512 | 89,45 s | `cauce/motion_maps/71_h3_warped_noise_safe_00001_.mp4` |
| H3 baseline seed/prompt matched | verificado | 124 f, 768×512 | 89,95 s | `cauce/motion_maps/71_h3_baseline_matched_00001_.mp4` |
| H3 latent warp + repair | decode válido; control pendiente de métrica | 124 f, 768×512, 20+12 steps | 153,55 s | `cauce/motion_maps/72_pass_1_00001_.mp4` + `72_pass_2_00001_.mp4` |

## Lectura de los resultados

El baseline FL2VA produjo video, latent estructural y receipt. Su audio generado
queda fuera del contrato productivo y los workflows actuales no lo decodifican
ni lo montan. Ref2VA también completó, pero referencias landscape sobre un output cuadrado
indujeron letterboxing visible. El segundo run `576×320` eliminó ese problema,
ocupó 72,6 s y produjo un MP4 de 1.361.003 bytes; ese profile queda verificado
para el envelope demo de 124 frames.

La primera continuación demostró que copiar y preservar 39 frames visuales es válido
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
  ├─ tail visual de 39 frames → target B + máscara preserve
  └─ último frame decodificado → FL2VA first_frame de B
```

El segundo camino fortalece el borde perceptual; el primero conserva historia
visual y fase causal. El run híbrido conservó la composición del río, el árbol
central, la perspectiva y la dirección de avance en el primer frame aceptado de
B, por lo que pasa este primer seam gate visual. Receipt:
`5d56dccb564f350687e10a5d8d760c219fe1a4ca30983d93910b0689db6a69e4`.

La primera implementación se ejecutó además con dos campos de color de 60 frames generados en
memoria, sin subir assets ni escribir un MP4 permanente. El build produjo el
dominio H3 esperado de 124 frames; `VAEEncode`, la inyección del video latent,
la máscara central, el sampler a 4 steps, el decode y el splice terminaron en
42,8 s. El preview final reportó 120 frames y el preview del parche 48, por lo
que la duración y el rango de reemplazo pasan el gate estructural. La calidad
temporal no se promueve hasta repetir con pares heterogéneos de gestos reales.

LanPaint 2.1.0 quedó instalado, habilitado y reiniciado correctamente en la
portable el 2026-08-22. La revisión posterior de su ruta de sampling mostró que
convierte `denoise_mask` a `(mask > 0.5)`: una curva continua en ese socket se
vuelve binaria. La tercera implementación separó por eso una región de sampling binaria con
12 frames de overscan por borde, una aceptación dura más pequeña y una opacidad
cosenoidal continua aplicada sólo después del decode.

La versión v3 se activó y ejecutó en la instancia real con dos clips H3 de 124
frames a 24 fps. El join resultante conserva los 248 frames originales
(10,333 s) y el artifact de diagnóstico contiene exactamente los 48 frames
aceptados (2 s). Fuera de ese rango, CAUCE no reemplaza ningún frame; los ocho
frames de cada borde mezclan el resultado decodificado con una curva cosenoidal.
En una medición luminancia-a-luminancia, el corte duro del par de prueba era un
pico aislado de 26,38; el frame equivalente del repair midió 20,68 dentro de un
tramo de movimiento sostenido, por lo que el salto dejó de ser una singularidad.
El intervalo generado introduce un zoom rápido por helechos entre ambos parents:
el contrato temporal y el splice pasan, pero la continuidad pedida falla. Ese
artifact queda rechazado y no se usa como evidencia de calidad.

La auditoría posterior identificó que el problema no era sólo el sampler. El core
0.33.1 precede las máscaras H3 por token y `MiniMaxH3AddGuide`; además v3 había
interpretado el segundo central como dos segundos y luego había abierto tres
segundos de sampling. La implementación validada genera sólo `[26,98)` y agrega
clips guía `[4,26)` y `[98,120)`. En el test de 3 s, la máscara temporal latente
fue `[0×8, 1×21, 0×8]`: denoise completo en 21 filas centrales y preservación
en las 16 filas exteriores. Se usaron 20 steps, sampler `res_multistep`,
scheduler `simple` y denoise `1.0` dentro de la máscara. El splice decodificado
aplicó cuatro frames de opacidad cosenoidal por borde. No hubo interpolación
temporal.

El resultado conservó los 248 frames originales (10,333 s). La medida de cambio
en el centro bajó de 26,38 para el corte duro a 6,04 en el temporal inpaint de
3 s; la inspección visual aprobó la continuidad del gesto para este par.

Durante la carga del ejemplo live se detectó además que Comfy serializa los
widgets ocultos `left_fps` y `right_fps` aunque estén conectados. El template
ahora conserva ambos valores antes de los parámetros de contexto para evitar
que `maximum_frames=362` se desplace al campo de segundos. Los nombres históricos
de artifacts se conservan sólo para trazabilidad; el workflow vigente escribe
bajo `cauce/demos/temporal_inpaint_*`.

Receipt Ref2VA landscape:
`357751bc96407eda0532dcc7b7a2b459c25838106a7a1a654f56daff219d6bc1`.

## Movimiento nativo sin referencia de video

Los workflows 71 y 72 se ejecutaron sin ningún MP4 de movimiento. El primero
intervino el ruido visual entregado a `SamplerCustomAdvanced`; el segundo
deformó directamente los 37 tokens visuales del parent H3 y realizó una pasada
de reparación. El stream estructural de audio permaneció copiado o con máscara
de generación cero y nunca se aceptó como audio de la pieza.

La primera calibración de ruido mostró el límite decisivo: `0,85` de
correlación temporal no equivale a una mezcla al 85 %, porque el anchor entra
con amplitud `sqrt(0,85)`. Junto al envelope de mapa `0,7`, la generación salió
del manifold visual aunque el job terminara sin error. Con `0,05` y `0,15`, el
decode volvió a ser coherente y conservó los endpoints. Por eso el workflow
publicado adopta esos valores como punto de partida y deja los valores altos
como stress/material failure cases.

La ruta de latent secuencial fue más tolerante: pass 1 generó el parent normal;
CAUCE aplicó un pan de 2 % y escala máxima 1,04 directamente sobre la grilla
latente causal; pass 2 reparó con 12 steps y denoise `0,35`. Ambos MP4 son
visualmente coherentes. Esto prueba que la geometría, el nested AV latent, la
máscara y el sampler están cableados correctamente, pero no demuestra todavía
que H3 reproduzca con precisión el campo pedido. La promoción requiere medir
optical-flow agreement y drift de endpoints contra el baseline matched.

## Transporte latente condicionado por sigma

El workflow 74 se ejecutó el 2026-08-23 sobre el core `924743af`, H3 FL2VA FP8,
Qwen3-VL NVFP4, 768×512, 124 frames, 20 steps `res_multistep`, seed
`2026082404` y un mapa affine `sine_loop` compartido. No entró ningún video de
movimiento. La primera implementación modificaba `x` antes del model call pero
dejaba `old_denoised` en la base espacial anterior. El job terminó en 94,8 s,
pero produjo bandas cromáticas horizontales; queda rechazado y documentado como
fallo de covarianza del historial de segundo orden.

CAUCE 0.9.1 transporta el estado actual y la predicción RES retenida con el
mismo incremento antes de cada evaluación. Un control con `strength=0` produjo
cuadros decodificados bit a bit idénticos al baseline en `t={0.1,1,2,3,4,5}`;
los contenedores MP4 difirieron sólo por metadata. Con `strength=0.10`, las
ventanas early `[0,0.45]`, middle `[0.25,0.75]` y late `[0.55,0.95]` completaron
sin bandas ni ruptura cromática visible. Los tiempos fueron 81,8 s baseline,
82,6 s early, 82,7 s middle y 82,7 s late.

Contra el baseline, la diferencia absoluta media por canal a 192×128 fue más
fuerte en early: aproximadamente 7,2 al inicio, 11,4 en el centro y 6,5 al
final. Middle llegó a 9,3 en el centro y late a 7,7. Esto confirma que la ventana
de sigma modifica la trayectoria aun con una fuerza pequeña, y que la misma
perturbación aplicada temprano tiene mayor alcance global. Es evidencia de
ejecución limpia y efecto no nulo; todavía no es una medición de fidelidad al
campo, que requiere optical flow y drift de endpoints.

## Sonda causal Euler `A →`

El 2026-08-24 se actualizó únicamente `ComfyUI-Cauce` al commit `65a1ae6` y se
reinició sólo el proceso de ComfyUI. El nodo actualizado cargó correctamente
sobre el mismo core, CUDA, PyTorch y modelos de la torre.

El control matched comparó Euler oficial con Euler envuelto por CAUCE a
`strength = 0`. Ambos outputs midieron `1.032.928` bytes y compartieron el
SHA-256 `ddfe9a631010e2119864117df01c68a5c05782ddab65cbe616505b85ea518e5f`:
la integración es bit a bit nula cuando el operador está apagado.

El primer preset causal era deliberadamente fuerte: traslación `0 → 32%`,
`strength = 0,25`, envelope acumulativo temprano y padding reflection. Su
desplazamiento retenido aproximado de `8%` produjo mosaicos y bandas cromáticas,
por lo que queda rechazado. Zoom y rotación decodificaron sin esa ruptura, pero
una ejecución limpia no demuestra obediencia geométrica.

Un segundo barrido horizontal matched usó Euler, padding border, strength
`0,10` y mapas `8%`, `16%` y `24%`, equivalentes aproximadamente a `0,8%`,
`1,6%` y `2,4%` retenidos. `0,8%` permaneció visualmente coherente; `1,6%` y
`2,4%` mostraron tearing creciente. La diferencia contra baseline creció con el
tiempo en la rama limpia, pero una búsqueda de registro horizontal a 192 px no
encontró desplazamiento direccional entero (`best dx = 0`). El estado correcto
es: runtime validado, región conservadora identificada, control de movimiento
todavía no demostrado.

## Reglas de promoción

- `verified`: ejecuta, produce artifacts completos y pasa inspección relevante.
- `executes`: el runtime termina, pero existe un problema de aspecto o calidad.
- `graph validated`: sockets y clases resuelven; todavía no hubo sample.
- `blocked`: falta una capability explícita; no se modifica el entorno para
  ocultarlo.
