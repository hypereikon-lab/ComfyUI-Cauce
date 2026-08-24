# Workflows de CAUCE

Esta suite convierte los nodos de CAUCE en recorridos de producción concretos.
Todos los graphs son workflows nativos de ComfyUI: no existe una interfaz
paralela, un editor externo ni una ontología de sujetos o acciones.

CAUCE organiza cuatro hechos: medios opacos, tiempo absoluto, relaciones entre
ventanas y artifacts versionados. La interpretación sigue ocurriendo en H3 a
partir de las imágenes, videos y prompts conectados. El soundtrack es un master
fijo: marca el reloj creativo y se monta al final, pero no se usa como target
generativo ni se reemplaza con audio sintetizado por H3.

## Preparación

1. Instalar o actualizar `ComfyUI-Cauce` desde Manager.
2. Reiniciar ComfyUI únicamente si cambió el código Python del pack.
3. Subir los archivos de `examples/assets/` al input de ComfyUI:
   - `cauce_forest_a.jpg`
   - `cauce_forest_b.jpg`
   - `cauce_forest_c.jpg`
   - `cauce_motion_reference.mp4`
4. Arrastrar un JSON de `examples/workflows/` sobre el canvas.
5. Ejecutar primero `00`, luego `10`; avanzar al resto cuando ese baseline esté
   estable.

Los workflows están fijados a los modelos existentes en la estación del lab:

```text
diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors
diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors
text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
vae/minimax_h3_video_vae_fp16.safetensors
vae/minimax_h3_audio_vae_fp32.safetensors
```

## Recorridos

| ID | Workflow | Ejecuta | Propósito |
|---|---|---:|---|
| 00 | `00_plate_sketch_handoff.json` | CPU | Componer un plate, asociarlo a tiempo/prompt y exportar PNG + sidecars. |
| 10 | `10_h3_fl2va_first_last.json` | 1 sample | Baseline first/last con MP4, parent latent y receipt. |
| 20 | `20_h3_ref2va_motion_reference.json` | 1 sample | Referencias visuales + video de movimiento; verificado landscape. |
| 30 | `30_h3_timed_guide.json` | 1 sample | Anclar una imagen intermedia; bloqueado en H3 0.33.1. |
| 40 | `40_h3_two_window_continuation.json` | 2 samples | Latent tail + endpoint visual; seam visual verificado. |
| 50 | `50_h3_temporal_inpainting.json` | 1 sample | Temporal inpainting localizado; verificado live con intervalo de 3 s. |
| 60 | `60_h3_native_latent_loop.json` | 4 samples | A→B→A con dos seams nativos sobre los AV latents originales. |
| 70 | `70_motion_map_composition.json` | 0 samples | Compara mapas compuestos con dos warps secuenciales de imagen. |
| 71 | `71_h3_warped_noise.json` | 1 sample | Campo advectado cerrado aplicado al ruido visual inicial de H3; preset conservador live-validado. |
| 72 | `72_h3_sequential_latent_pass.json` | 2 samples | Generación base + warp del latent H3 + segunda pasada a denoise 0,35; decode live-validado. |
| 73 | `73_depth_advection_preview.json` | 0 samples | Reproyección 2.5D + advección compuestas con validity de disoclusiones. |
| 90 | `90_storage_maintenance.json` | CPU | Inventario físico y limpieza en dos fases de `input/` u `output/`. |

Los recorridos 70–73 forman una matriz experimental, no cuatro presets de
producción. Ejecutar primero 70 y 73 permite descartar mapas inválidos sin usar
la GPU generativa. 71 prueba control desde el ruido; 72 prueba una intervención
nativa y secuencial sobre el latent. La comparación completa y sus criterios
están en [`MOTION_MAPS.md`](MOTION_MAPS.md).

En la validación del lab, 71 sólo permaneció dentro del manifold visual con
señal conservadora (`temporal_correlation=0,05`, envelope `0,15`). El stress
test `0,85`/`0,7` ejecutó pero produjo un decode cromáticamente corrupto. El
workflow 72 completó ambas pasadas y mantuvo un decode coherente después de un
pan de 2 %, escala 1,04 y denoise 0,35. Estos resultados validan la ruta tensorial;
la fidelidad del movimiento pretendido todavía debe medirse contra el baseline
con seed, prompt y endpoints idénticos.

Los workflows visuales 00–50 fueron cargados y resueltos por el frontend real
de la instancia del lab: ningún graph contiene nodos desconocidos ni depende de
los packs legacy. El workflow 90 se valida localmente antes de su activación en
el lab. Los resultados ejecutados, sus tiempos y los gates pendientes están en
[`LAB_RESULTS.md`](LAB_RESULTS.md). “Graph validado” no equivale a “calidad
visual aprobada”.

## 00 · Plate sketch y handoff

El graph de plates usa tres medios opacos como ejemplo. Pueden reemplazarse por
cualquier imagen, máscara o composición intermedia. Los nodos de layer exponen
posición, escala, rotación, opacidad, blend y feather; no intentan reconocer lo
que hay en cada capa.

`CAUCE · Export Plate` guarda:

```text
ComfyUI/output/cauce/plates/<point>_<timecode>_plate_*.png
ComfyUI/output/cauce/plates/<mismo_nombre>.prompt.txt
ComfyUI/output/cauce/plates/<mismo_nombre>.point.json
```

El PNG y el prompt son el handoff directo a Runway/GPT Image. Al regresar un
resultado, se reemplaza o adjunta la imagen en el mismo punto sin cambiar el
reloj ni el resto del workflow.

## 10 · FL2VA first/last

Éste es el baseline de producción. Hace explícitos:

```text
first IMAGE + last IMAGE + arbitrary prompt
  → CAUCE generation window
  → official MiniMaxH3ImageToVideo
  → H3 sigma shifts internos
  → res_multistep + simple / 20 steps
  → resolved phase-safe parent
  → video decode; audio H3 descartado
  → accepted temporal range
  → MP4 visual + latent estructural + receipt
```

Para iterar normalmente sólo se cambian imágenes, prompt y seed. El profile
`h3-5090-fl2va-640` mantiene el primer barrido dentro del envelope medido de la
5090 de 32 GB. No subir resolución o duración y número de referencias al mismo
tiempo: cambiar una sola dimensión deja receipts comparables.

## 20 · Ref2VA como campo de referencias

El demo conecta dos imágenes y un video de 56 frames a 24 fps. Los tags que H3
recibe son:

```text
<Picture 1>, <Picture 2>, <Video 1>
```

El prompt decide qué función cumple cada referencia. CAUCE conserva el orden,
valida cantidades y duración, y entrega los medios a Ref2VA; no almacena una
clasificación “identidad”, “estilo”, “movimiento” o “cámara”. El mismo socket de
video puede recibir primitivas, simulaciones, render 3D, optical flow
visualizado o material filmado.

El MP4 de demo dura 2,333 s y contiene 56 frames, una longitud válida `17k+5`.
Para reemplazarlo, el video debe estar a 24 fps y durar entre 2 y 15 s. El total
temporal de referencias video no puede exceder 15 s en este preset. Los sockets
de audio existen por compatibilidad upstream, pero no se usan en producción.

El workflow usa `h3-5090-ref2va-576x320`: mantiene un costo cercano al profile
cuadrado de 448, pero ofrece un canvas landscape más apropiado para referencias
panorámicas. El profile 448 permanece disponible para material realmente
cuadrado; cambiar de aspect ratio es una decisión de output, no una propiedad
semántica de las referencias.

## 30 · Guide en tiempo absoluto

`CAUCE · H3 Timed Guide` resuelve `master_seconds` contra el inicio real de la
ventana y llama al `MiniMaxH3AddGuide` oficial. Esto permite que un frame o clip
esté anclado a la música o a un punto de montaje sin codificar a mano un índice
local del latent. El runtime `ComfyUI 0.33.1` actualmente instalado en
el lab todavía no registra `MiniMaxH3AddGuide`; por eso este workflow carga y se
puede inspeccionar, pero queda bloqueado con un error explícito hasta que esa
clase oficial esté disponible. FL2VA y Ref2VA no dependen de ella.

Se pueden encadenar varios guides. Cada uno debe caber completamente en la
ventana renderizada. Para un clip guía, usar una longitud H3 válida. El master
de audio no se conecta a AddGuide.

## 40 · Continuación de dos ventanas

La segunda ventana declara `context_frames = 39`. El graph:

1. genera y resuelve el parent A;
2. decodifica A y selecciona su último frame aceptado de forma opaca;
3. usa ese endpoint como `first_frame` oficial de FL2VA para B;
4. copia exactamente el tail visual de A al head del target B;
5. coloca máscara cero sobre esa región heredada;
6. congela el stream de audio estructural y genera sólo video desconocido;
7. acepta y guarda A y B por separado.

La ventana B solicita `85` frames nuevos (`3,541666667 s`). Con los `39`
heredados, el render vuelve a sumar `124` frames: el mismo envelope ya verificado
para el baseline. Esto evita que la demostración de continuidad cambie duración
y algoritmo al mismo tiempo.

No se concatenan latents H3 independientes antes del VAE. Cada parent se
decodifica dentro de su propia fase causal y sólo las regiones aceptadas se
montan después sobre el reloj maestro.

`39` permanece como preset ya probado, pero ya no es una obligación AV. Los
boundaries visuales legales son `5`, `22`, `39`, `56`, etc.; esto permite elegir
contexto exclusivamente por continuidad visual.

En el runtime 0.33.1, el latent tail por sí solo ejecutó pero produjo un salto
visual fuerte. La guía de clip latente pertenece a un core H3 posterior y CAUCE
la bloquea explícitamente cuando no está disponible. El endpoint decodificado
es la estrategia compatible que ahora debe pasar el gate comparativo; no se
declara continuidad “resuelta” sólo porque el sampler termine. En la prueba
del lab, la variante híbrida conservó composición, perspectiva, cauce y árbol
central a través del join. El audio generado no forma parte de ese gate ni se
acepta en la pieza.

## 50 · Temporal inpainting localizado

Este recorrido parte de dos videos ya existentes, no de parents latentes. Los
normaliza al profile FL2VA 640, toma 2,5 s de contexto a cada lado y entrega el
tail de A y el head de B al VAE como un único dominio causal. El nodo de build
produce también la ventana H3 exacta.

Con los defaults:

```text
A: últimos 60 frames ┐
                     ├─ + 2 guards por lado → working domain de 124 frames
B: primeros 60 frames┘

corte working: frame 62
clip guía izquierdo:  [4, 26) = 22 frames preservados
intervalo generable:  [26, 98) = 72 frames = 3 s
clip guía derecho:    [98, 120) = 22 frames preservados
output: len(A) + len(B), sin cambio de duración
```

Los tres segundos pedidos no se duplican por lado: son 72 frames en total,
36 de la cola de A y 36 del inicio de B. Los límites coinciden con la grilla
temporal de H3 para que ningún token quede parcialmente generado. `CAUCE · H3
Temporal Guide Clips` agrega los dos clips preservados al conditioning oficial; el modelo ve
el movimiento de entrada y de salida, además de first/last y el prompt
arbitrario.

`CAUCE · Temporal Inpaint Fields` produce tres `MASK` inspeccionables:

```text
sampling_support     → soporte binario exacto por token H3
hard_acceptance      → intervalo binario permitido
output_opacity       → mezcla decodificada del parche
```

Después del decode, `output_opacity` usa una curva cosenoidal continua de cuatro
frames. Ésta suaviza el splice visible; no representa una fuerza de denoise
fraccional.

La auditoría de las implementaciones rechazadas encontró dos causas
estructurales. Primero, el gráfico
interpretaba 1 s como 1 s por lado y añadía 0,5 s de overscan por lado: H3
inventaba 72 frames/3 s. Segundo, ComfyUI v0.33.1 es anterior a la
implementación oficial que entrega la máscara por token al modelo, etiqueta
cada fila con su timestep correcto e inyecta el latent preservado a fuerza de
conditioning. El sampler podía completar sin que el edit fuese válido.

La implementación vigente usa `SamplerCustomAdvanced` oficial y comprueba en runtime:

```text
MiniMaxH3AddGuide
MiniMaxH3._token_grid_masks
MiniMaxH3._denoise_mask_conds
MiniMaxH3.scale_latent_inpaint
```

Si falta cualquiera, falla antes del sample y solicita actualizar el core
oficial. La actualización requerida no implica cambiar CUDA, PyTorch ni los
modelos.

Después del decode, CAUCE acepta sólo el parche central y repone el resto desde
los inputs originales. Modificar el exterior no es una solicitud al modelo,
sino una imposibilidad topológica.

El JSON contiene `gesture_a.mp4` y `gesture_b.mp4` como placeholders. Cada clip
debe tener al menos 60 frames y ambos deben ser exactamente 24 fps. Para cada
evaluación se mantiene fijo el source, seed, prompt, intervalo y clips guía; se
varía una sola variable por vez.
La resolución queda normalizada a 640×640 para el envelope verificado de la
5090.

Las versiones anteriores confirmaron sockets, shapes, decode y splice, pero la
prueba perceptual fue mala. La implementación vigente superó el par de gestos
con un intervalo de 3 s; un job exitoso sigue sin bastar para promover nuevos
pares de material.

El workflow es intencionalmente video-only. El audio master no se conecta, no
se codifica y no se sustituye. H3 mantiene internamente un stream de audio vacío
y enmascarado sólo porque su latent es estructuralmente AV; ese resultado se
descarta. Para una pieza larga se corrigen clips locales y luego se colocan
sobre el reloj del master fijo.

## 60 · Loop FL2VA con dos seams nativos

Este recorrido genera A→B y B→A bajo el mismo profile landscape
`h3-5090-fl2va-768x512`. Guarda los dos AV latents finales antes de decodificar
y los usa directamente como contexto para dos operaciones distintas:

```text
seam B: tail latent de A→B + centro generado + head latent de B→A
seam A: tail latent de B→A + centro generado + head latent de A→B
```

No concatena latents independientes ni vuelve a codificar los MP4. Para el
preset de 124 frames, la geometría visible y latent es:

```text
video visible: 22 protegidos + 80 muestreados + 22 protegidos = 124
video latent:   7 protegidos + 23 generados + 7 protegidos = 37 tokens
rangos target: [0,7) preserve · [7,30) generate · [30,37) preserve
```

El token 30 vuelve a fase cero en el ciclo temporal `(1,4,4,4,4)` del VAE H3.
Por eso el head del segundo parent puede ocupar `[30,37)` sin reinterpretar la
duración visible de sus filas. El tail del primer parent también comienza en
token 30 para un source de 124 frames. CAUCE rechaza automáticamente longitudes
o combinaciones que rompan esta igualdad de fase.

Cada sample de seam se decodifica como una ventana de inspección de 124 frames,
pero el montaje sólo acepta `[26,98)`: los 72 frames internos (tres segundos).
Los cuatro frames de overscan a cada lado se muestrean, pero no se montan. Dos
guide clips de 22 frames, `[0,22)` y `[102,124)`, exponen explícitamente el gesto
entrante y saliente al conditioning. El patch aceptado reemplaza 36 frames de
cada clip. Un feather coseno de cuatro frames opera después del decode; no
cambia el mask de denoise ni permite que H3 reescriba el contexto protegido.

El montaje final conserva exactamente:

```text
124 frames A→B reparados + 124 frames B→A reparados
= 248 frames / 10,333 s a 24 fps
```

El seam B queda dentro del MP4. El seam A se divide entre el final y el primer
frame del archivo, por lo que se evalúa reproduciendo el output en loop. El
audio H3 se congela y descarta; el master fijo se monta posteriormente.

## 90 · Mantenimiento de storage

Este workflow no usa GPU. `Storage Inventory` inspecciona físicamente una raíz
de Comfy y produce un `CAUCE_STORAGE_PLAN`, un report JSON, el número de
archivos, los GiB lógicos y un código de confirmación. Con `root = input` puede
encontrar uploads aunque Assets no los muestre; con `root = output` incluye
generaciones antiguas o no indexadas.

La primera corrida siempre se hace con `armed = false`. Cleanup recibe el plan
y el código por links, permanece inerte y guarda ese plan exacto fuera de las
raíces limpiables. Tras revisar el report se cambia únicamente `armed = true` y
se vuelve a ejecutar. Un plan nuevo o modificado que no haya pasado por la
corrida desarmada se rechaza. Cada archivo se valida contra
el plan inmediatamente antes de borrarse. Los archivos nuevos o modificados se
omiten; las carpetas vacías sólo se retiran si eran padres de archivos realmente
eliminados. El receipt queda bajo `user/cauce/maintenance/receipts/`, fuera de
la limpieza.

Defaults para vaciar una raíz conservando markers:

```text
relative_subfolder = .
include_glob = *
exclude_glob =
recursive = true
minimum_age_minutes = 0
preserve_markers = true
```

## Runner: secuencia reiniciable

Los JSON visuales sirven para diseño e inspección. Para dejar varias ventanas
corriendo, CAUCE incluye dos plantillas API:

```text
examples/api/h3_fl2va_window.template.json
examples/api/h3_continuation_window.template.json
```

`examples/project.example.json` usa la primera para el parent inicial y la
segunda para una continuación. Cada ventana puede seleccionar su propia
`workflow_template`; el runner materializa el JSON exacto, escribe estado antes
y después de cada submit, y omite ventanas completas al reanudar.

```bash
python cauce_cli.py resume examples/project.example.json --dry-run
python cauce_cli.py resume examples/project.example.json
python cauce_cli.py status examples/project.example.json
```

La plantilla de continuación carga el parent guardado, lo decodifica y deriva
automáticamente su `first_frame`; no exige exportar manualmente un PNG de borde.

Los assets nombrados por una plantilla API deben existir previamente en
`ComfyUI/input/`. El modo recomendado es ejecutar el runner en la torre contra
`http://127.0.0.1:8188`; el login interactivo del túnel no se reutiliza.

## Ciclo operativo

```text
plate sketch
  → export PNG + prompt
  → generación de imagen en Runway
  → resultado vuelve a un CAUCE point
  → FL2VA / Ref2VA / guide
  → accepted MP4 + phase-safe parent + receipt
  → aprobar, rerollear o continuar
  → ensamblar regiones aceptadas en el master clock
```

El receipt fija seed, profile, sampler, scheduler, steps, hashes de modelos,
parents y hash del workflow. Un reroll puede reemplazar el índice del artifact
sin perder qué parent produjo la rama anterior.

## Fallos esperables

- **Modelo no aparece:** ejecutar `CAUCE · Preflight`; no instalar otra variante
  hasta revisar espacio y manifest.
- **Asset rojo o ausente:** subirlo primero a `ComfyUI/input/` y volver a
  seleccionarlo en `Load Image`/`Load Video`.
- **Out of memory:** volver al profile 640 FL2VA o 448 Ref2VA, cerrar previews
  pesados y reiniciar únicamente ComfyUI si la VRAM quedó fragmentada.
- **Ref2VA rechaza el video:** comprobar 24 fps, 2–15 s y duración acumulada.
- **Guide fuera de rango:** revisar el `render_range`, no sólo el tiempo visible
  solicitado.
- **Continuation antes de cero:** el tiempo aceptado debe dejar espacio para el
  context heredado.
- **`mask_plus_guide` no disponible:** usar el workflow 40 con endpoint
  decodificado; no actualizar ComfyUI automáticamente desde CAUCE.
- **Temporal inpainting rechaza el input:** ambos videos deben ser 24 fps, tener al menos
  el contexto solicitado y normalizarse al mismo profile antes del VAE.
- **Temporal inpainting ejecuta pero no corrige el gesto:** variar primero seed/prompt;
  si el nodo reporta runtime incompleto, actualizar primero el core oficial;
  luego comparar clips guía de 22/39 frames sin cambiar a la vez centro,
  resolución y seed.
- **El túnel cae:** el proceso de Comfy puede continuar en la torre; consultar
  history al reconectar antes de volver a encolar.
